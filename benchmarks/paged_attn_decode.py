# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import argparse
import itertools

import torch
from triton.testing import do_bench  # @manual

max_seq_len = 128 * 1024
B_list = [1, 96, 192]
kv_len_list = [l for l in range(8192, 32 * 1024 + 1, 8192)] + [max_seq_len]
num_heads = 64
num_kv_heads = 8
head_dim = 64
logit_cap = 0.0
kv_cahce_type_list = ["bf16", "fp8"]
q_dtype_list = ["bf16", "fp8"]
fp8_dtype = torch.float8_e4m3fnuz if torch.version.hip else torch.float8_e4m3fn

FLASHINFER_WORKSPACE_BUFFER_SIZE = 256 * 1024 * 1024


def compute_bytes_moved(b, kv_len, one_max_len, kv_cache_dtype, q_dtype):
    bytes_per_kv_ele = 1 if kv_cache_dtype == "fp8" else 2
    bytes_per_q_ele = 1 if q_dtype == "fp8" else 2
    bytes_per_token = num_kv_heads * head_dim * bytes_per_kv_ele
    if not one_max_len:
        bytes_total = 1.0 * b * kv_len * bytes_per_token
    else:
        bytes_total = 1.0 * ((b - 1) * kv_len + max_seq_len) * bytes_per_token
    bytes_total += bytes_per_q_ele * b * num_heads * head_dim * bytes_per_token
    return bytes_total / 1e9


def construct_attention_inputs(
    b, kv_len, q_dtype, kv_cache_dtype, one_max_len, page_size
):
    device = "cuda"
    total_page = b * max_seq_len // page_size + 1
    q = torch.randn((b, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    kv_cache = torch.randn(
        (total_page, 2, page_size, num_kv_heads, head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    if kv_cache_dtype == "fp8":
        kv_cache = kv_cache.to(fp8_dtype)

    o = torch.empty((b, num_heads, head_dim), dtype=torch.bfloat16, device=device)
    if q_dtype == "fp8":
        q = q.to(fp8_dtype)
        o = o.to(fp8_dtype)
    cu_seqlens_q = torch.arange(0, b + 1, dtype=torch.int32, device=device)
    max_seqlen_q = 1
    if one_max_len:
        seqused_k = torch.tensor(
            [kv_len] * (b - 1) + [max_seq_len], dtype=torch.int32, device=device
        )
        max_seqlen_k = max_seq_len
    else:
        seqused_k = torch.tensor([kv_len] * b, dtype=torch.int32, device=device)
        max_seqlen_k = kv_len

    block_table = torch.arange(1, total_page, dtype=torch.int32, device=device).reshape(
        b, max_seq_len // page_size
    )
    return (
        q,
        kv_cache,
        o,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        block_table,
    )


def bench_triton_unified_attention(q_dtype, kv_cache_dtype, b, kv_len, one_max_len):
    from vllm.attention.ops.triton_unified_attention import unified_attention

    page_size = 32
    q, kv_cache, o, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k, block_table = (
        construct_attention_inputs(
            b, kv_len, q_dtype, kv_cache_dtype, one_max_len, page_size
        )
    )
    k, v = kv_cache.unbind(1)

    softmax_scale = 0.125
    causal = True
    window_size = (-1, -1)
    softcap = 0.0
    q_descale = None
    k_descale = torch.tensor([1.0] * num_kv_heads, dtype=torch.float32, device=q.device)
    v_descale = torch.tensor([1.0] * num_kv_heads, dtype=torch.float32, device=q.device)
    alibi_slopes = None
    qq_bias = None
    sinks = torch.randn(num_heads, dtype=torch.bfloat16, device=q.device)

    bench_fn = lambda: unified_attention(
        q,
        k,
        v,
        o,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        causal,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        alibi_slopes=alibi_slopes,
        qq_bias=qq_bias,
        sinks=sinks,
    )
    t_us = do_bench(bench_fn) * 1000.0
    gb_moved = compute_bytes_moved(b, kv_len, one_max_len, kv_cache_dtype, q_dtype)
    gb_per_sec = gb_moved / (t_us / 1e6)
    print(
        f"unified_triton_attn, {b}, {kv_len}, {one_max_len}, {q_dtype}, {kv_cache_dtype}, {t_us}, {gb_per_sec}"
    )


def bench_trtllm_decode_attention(q_dtype, kv_cache_dtype, b, kv_len, one_max_len):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache

    page_size = 16
    q, kv_cache, o, _, _, seqused_k, max_seqlen_k, block_table = (
        construct_attention_inputs(
            b, kv_len, q_dtype, kv_cache_dtype, one_max_len, page_size
        )
    )

    flashinfer_workspace_buffer = torch.zeros(
        FLASHINFER_WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device=q.device
    )
    kv_cache_permuate = kv_cache.permute(0, 1, 3, 2, 4)
    sinks = torch.randn(num_heads, dtype=torch.float32, device=q.device)
    bmm1_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    bmm2_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    bench_fn = lambda: trtllm_batch_decode_with_kv_cache(
        query=q,
        kv_cache=kv_cache_permuate,
        workspace_buffer=flashinfer_workspace_buffer,
        block_tables=block_table,
        seq_lens=seqused_k,
        max_seq_len=max_seqlen_k,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        window_left=-1,
        sinks=sinks,
        o_sf_scale=None,
        out=o,
    )

    t_us = do_bench(bench_fn) * 1000.0
    gb_moved = compute_bytes_moved(b, kv_len, one_max_len, kv_cache_dtype, q_dtype)
    gb_per_sec = gb_moved / (t_us / 1e6)
    print(
        f"trtllm, {b}, {kv_len}, {one_max_len}, {q_dtype}, {kv_cache_dtype}, {t_us}, {gb_per_sec}"
    )


def bench_aiter_decode_attention(q_dtype, kv_cache_dtype, b, kv_len, one_max_len=False):
    page_size = 16
    (
        query,
        kv_cache,
        out,
        cu_seqlens_q,
        _,
        seqused_k,
        max_seqlen_k,
        block_table,
    ) = construct_attention_inputs(
        b, kv_len, q_dtype, kv_cache_dtype, one_max_len, page_size
    )

    if kv_cache_dtype == "fp8":
        kv_cache = kv_cache.view(torch.uint8)
    key_cache, value_cache = kv_cache.unbind(1)

    partition = 256

    max_num_partitions = (max_seqlen_k + partition - 1) // partition

    nbytes_per_qo_elem = 2
    workspace_buffer = torch.empty(
        (b * num_heads * max_num_partitions * head_dim) * nbytes_per_qo_elem
        + 2 * (b * num_heads * max_num_partitions) * 4,
        dtype=torch.uint8,
        device=out.device,
    )

    k_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)
    v_scale = torch.tensor(1.0, dtype=torch.float32, device=query.device)

    bench_fn = lambda: torch.ops.aiter.paged_attention_v1(
        out,
        workspace_buffer,
        query,
        key_cache,
        value_cache,
        0.08838834764831845,
        block_table,
        cu_seqlens_q,
        seqused_k,
        max_seqlen_k,
        None,
        "fp8",
        "NHD",
        0,
        k_scale,
        v_scale,
        None,
        partition,
    )
    t_us = do_bench(bench_fn) * 1000.0
    gb_moved = compute_bytes_moved(b, kv_len, one_max_len, kv_cache_dtype, q_dtype)
    gb_per_sec = gb_moved / (t_us / 1e6)
    print(
        f"aiter, {b}, {kv_len}, {one_max_len}, {kv_cache_dtype}, {t_us}, {gb_per_sec}"
    )


def bench_decode_attention(backend, q_dtype, kv_cache_dtype, b, kv_len, one_max_len):
    if b == 1 and one_max_len:
        return
    if q_dtype == "fp8" and kv_cache_dtype == "bf16":
        return
    if q_dtype == "fp8" and backend in ["triton", "aiter"]:
        return

    kwargs = {
        "q_dtype": q_dtype,
        "kv_cache_dtype": kv_cache_dtype,
        "b": b,
        "kv_len": kv_len,
        "one_max_len": one_max_len,
    }
    if backend == "triton":
        bench_triton_unified_attention(**kwargs)
    elif backend == "aiter":
        bench_aiter_decode_attention(**kwargs)
    elif backend == "trtllm":
        bench_trtllm_decode_attention(**kwargs)
    else:
        raise ValueError(f"Unknown backend {backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode Paged Attention Benchmarking")
    parser.add_argument("--backend", type=str, default="trtllm")
    args = parser.parse_args()
    for b, kv_len, one_max_len, q_dtype, kv_cache_dtype in itertools.product(
        B_list, kv_len_list, [False, True], q_dtype_list, kv_cahce_type_list
    ):
        bench_decode_attention(
            args.backend, q_dtype, kv_cache_dtype, b, kv_len, one_max_len
        )
