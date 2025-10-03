# pyre-unsafe
import argparse
import itertools
from dataclasses import dataclass

import torch

from triton.testing import do_bench  # @manual
from vllm.attention.ops.triton_decode_attention import decode_attention_fwd_grouped
from vllm.platforms import current_platform


@dataclass
class BenchConfig:
    backend: str
    num_q_heads: int
    kv_cache_dtype: str
    b: int
    kv_len: int
    one_max_len: bool


B_list = [32, 64, 128, 192]
kv_len_list = [2048] + [l for l in range(4096, 32 * 1024 + 1, 4096)]
split_kv_list = [4, 5, 6, 7, 8, 9, 10, 11, 12]
kv_lora_rank = 512
qk_rope_head_dim = 64
qk_dim = kv_lora_rank + qk_rope_head_dim
sm_scale = 0.13086079996295005
logit_cap = 0.0
max_seq_len = 128 * 1024

kv_cahce_type_list = ["bf16", "fp8"]
page_size = 32

fp8_dtype = current_platform.fp8_dtype()


def compute_bytes_moved(bc):
    total_bytes = 0
    num_bytes_per_elem = 2 if bc.kv_cache_dtype == "bf16" else 1
    total_bytes += bc.b * bc.num_q_heads * qk_dim * num_bytes_per_elem
    if bc.one_max_len:
        total_kv_len = max_seq_len + (bc.b - 1) * bc.kv_len
    else:
        total_kv_len = bc.b * bc.kv_len
    total_bytes += total_kv_len * qk_dim * num_bytes_per_elem
    total_bytes += bc.b * bc.num_q_heads * kv_lora_rank * num_bytes_per_elem
    return total_bytes / 1e9


def construct_mla_decode_attention_inputs(bc):
    device = "cuda"
    q = torch.randn((bc.b, bc.num_q_heads, qk_dim), dtype=torch.bfloat16, device=device)
    kv_c_and_k_pe_cache = torch.randn(
        (bc.b * max_seq_len // page_size, page_size, qk_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    if bc.kv_cache_dtype == "fp8":
        q = q.to(fp8_dtype)
        kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.to(fp8_dtype)

    o = torch.empty((bc.b, bc.num_q_heads, kv_lora_rank), dtype=q.dtype, device=device)
    page_table_cpu = []
    page_id = 0
    for _ in range(bc.b):
        page_table_row = []
        for _ in range(bc.kv_len // page_size):
            page_table_row.append(page_id)
            page_id += 1
        page_table_cpu.append(page_table_row)
    page_table = torch.tensor(page_table_cpu, dtype=torch.int32, device=device)
    if one_max_len:
        k_seq_len = torch.tensor(
            [kv_len] * (b - 1) + [max_seq_len], dtype=torch.int32, device=device
        )
    else:
        k_seq_len = torch.tensor([bc.kv_len] * bc.b, dtype=torch.int32, device=device)
    return (
        q,
        kv_c_and_k_pe_cache,
        o,
        page_table,
        k_seq_len,
    )


def bench_and_print(bench_fn, bc):
    t_us = do_bench(bench_fn) * 1000.0
    gb_moved = compute_bytes_moved(bc)
    gb_per_sec = gb_moved / (t_us / 1e6)

    print(
        f"{bc.backend}, {bc.num_q_heads}, {bc.b}, {bc.kv_len}, {bc.kv_cache_dtype}, {t_us}, {gb_per_sec}"
    )


def bench_triton_mla_decode_attention(bc):
    (
        q,
        kv_c_and_k_pe_cache,
        o,
        page_table,
        k_seq_len,
    ) = construct_mla_decode_attention_inputs(bc)
    kv_c_and_k_pe_cache = kv_c_and_k_pe_cache.unsqueeze(2)
    lse = torch.zeros(bc.b, bc.num_q_heads, dtype=q.dtype, device=q.device)
    num_kv_splits = 4

    attn_logits = torch.empty(
        (
            bc.b,
            bc.num_q_heads,
            num_kv_splits,
            kv_lora_rank + 1,
        ),
        dtype=torch.float32,
        device=q.device,
    )
    bench_fn = lambda: decode_attention_fwd_grouped(
        q,
        kv_c_and_k_pe_cache,
        kv_c_and_k_pe_cache[..., :kv_lora_rank],
        o,
        lse,
        page_table,
        k_seq_len,
        attn_logits,
        num_kv_splits,
        sm_scale,
        page_size,
        logit_cap,
    )
    bench_and_print(bench_fn, bc)


def bench_cutlass_mla_decode_attention(bc):
    import vllm._custom_ops as ops

    (
        q,
        kv_c_and_k_pe_cache,
        o,
        page_table,
        k_seq_len,
    ) = construct_mla_decode_attention_inputs(bc)
    initial_workspace_size = 128 * 1024 * 1024
    workspace_buf = torch.empty(
        initial_workspace_size, device=q.device, dtype=torch.uint8
    )
    q_nope, q_pe = torch.split(q, [kv_lora_rank, qk_dim - kv_lora_rank], dim=-1)
    lse = torch.Tensor()
    num_kv_splits = -1

    bench_fn = lambda: ops.sm100_cutlass_mla_decode(
        o,
        lse,
        q_nope,
        q_pe,
        kv_c_and_k_pe_cache,
        k_seq_len,
        page_table,
        workspace_buf,
        sm_scale,
        num_kv_splits,
    )
    bench_and_print(bench_fn, bc)


def bench_trtllm_mla_decode_attention(bc):
    from flashinfer.decode import trtllm_batch_decode_with_kv_cache_mla

    (
        q,
        kv_c_and_k_pe_cache,
        o,
        page_table,
        k_seq_len,
    ) = construct_mla_decode_attention_inputs(bc)
    o = o.to(torch.bfloat16)
    max_seqlen_k = bc.kv_len if not bc.one_max_len else max_seq_len

    FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
    g_fi_workspace = torch.zeros(
        FLASHINFER_MLA_WORKSPACE_BUFFER_SIZE,
        dtype=torch.uint8,
        device="cuda",
    )
    qk_nope_head_dim = 128
    q = q.unsqueeze(1)
    bmm1_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)
    bmm2_scale = torch.tensor(1.0, dtype=torch.float32, device=q.device)

    bench_fn = lambda: trtllm_batch_decode_with_kv_cache_mla(
        query=q,
        kv_cache=kv_c_and_k_pe_cache.unsqueeze(1),
        workspace_buffer=g_fi_workspace,
        qk_nope_head_dim=qk_nope_head_dim,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=page_table,
        seq_lens=k_seq_len,
        max_seq_len=max_seqlen_k,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        out=o,
    )
    bench_and_print(bench_fn, bc)


def bench_aiter_mla_decode_attention(bc):
    page_size = 1
    from vllm.attention.ops.rocm_aiter_mla import aiter_mla_decode_fwd

    (
        q,
        kv_c_and_k_pe_cache,
        o,
        page_table,
        k_seq_len,
    ) = construct_mla_decode_attention_inputs(bc)

    kv_buffer = kv_c_and_k_pe_cache.unsqueeze(2)
    qo_indptr = torch.arange(0, bc.b + 1, step=1, dtype=torch.int32, device=q.device)
    block_table_bounds = (k_seq_len + page_size - 1) // page_size
    paged_kv_indptr = torch.cat(
        [
            torch.zeros(1, dtype=block_table_bounds.dtype, device=q.device),
            block_table_bounds.cumsum(dim=0, dtype=torch.int32),
        ]
    )
    mask = torch.arange(
        page_table.size(1), dtype=page_table.dtype, device=q.device
    ).unsqueeze(0) < block_table_bounds.unsqueeze(1)
    paged_kv_indices = page_table[mask]
    paged_kv_last_page_len = k_seq_len % page_size
    paged_kv_last_page_len = torch.where(
        paged_kv_last_page_len == 0, page_size, paged_kv_last_page_len
    )

    # max_seqlen_qo must be 1 except for MTP
    # TODO: Find the best value for MTP
    max_seqlen_qo = 1
    bench_fn = lambda: aiter_mla_decode_fwd(
        q,
        kv_buffer,
        o,
        sm_scale,
        qo_indptr,
        max_seqlen_qo,
        paged_kv_indptr,
        paged_kv_indices,
        paged_kv_last_page_len,
    )
    bench_and_print(bench_fn, bc)


def bench_mla_decode_attention(bc):
    if bc.b == 1 and bc.one_max_len:
        return
    if bc.kv_cache_dtype == "fp8" and bc.backend == "aiter":
        return

    if bc.backend == "trtllm":
        bench_trtllm_mla_decode_attention(bc)
    elif bc.backend == "triton":
        bench_triton_mla_decode_attention(bc)
    elif bc.backend == "cutlass":
        bench_cutlass_mla_decode_attention(bc)
    elif bc.backend == "aiter":
        bench_aiter_mla_decode_attention(bc)
    else:
        raise ValueError(f"Unknown backend {bc.backend}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode MLA Attention Benchmarking")
    parser.add_argument("--backend", type=str, default="triton")
    parser.add_argument("--num_heads", type=int, default=128)
    args = parser.parse_args()
    for b, kv_len, one_max_len, kv_cache_dtype in itertools.product(
        B_list, kv_len_list, [False], kv_cahce_type_list
    ):
        bc = BenchConfig(
            backend=args.backend,
            num_q_heads=args.num_heads,
            kv_cache_dtype=kv_cache_dtype,
            b=b,
            kv_len=kv_len,
            one_max_len=one_max_len,
        )
        bench_mla_decode_attention(bc)
