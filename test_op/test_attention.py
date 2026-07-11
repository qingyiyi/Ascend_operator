import os
import sys

import numpy as np
import torch
import torch_npu
from diff_tensor import compute_diff, print_diff


TEST_OP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(TEST_OP_DIR)
EXTENSION_DIR = os.path.join(
    PROJECT_ROOT,
    "custom_ops",
    "PagedAttentionMixV3",
    "torch_extension",
)
if EXTENSION_DIR not in sys.path:
    sys.path.insert(0, EXTENSION_DIR)

import paged_attention_mix_v3_ext  # noqa: F401,E402


np.set_printoptions(threshold=np.inf, suppress=True)
torch.set_printoptions(
    precision=8,
    threshold=1_000_000,
    edgeitems=1_000_000,
    linewidth=1_000_000,
    sci_mode=False,
)


def profile_targets():
    target = os.getenv("PROFILE_TARGET", "").lower()
    if target in ("none", "null", "off", "false", "0"):
        return target, False, False
    profile_mix_v3 = target in ("mix_v3", "both") or os.getenv("PROFILE_MIX_V3", "0") == "1"
    profile_atb = target in ("atb", "splitfuse", "both") or os.getenv("PROFILE_ATB", "0") == "1"
    return target, profile_mix_v3, profile_atb


def is_profile_enabled():
    _, profile_mix_v3, profile_atb = profile_targets()
    return profile_mix_v3 or profile_atb


class NpuProfile:
    def __init__(
        self,
        *,
        chrome_trace=None,
        tensorboard_trace=None,
        profiler_level=0,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        enable=True,
    ):
        self.chrome_trace = chrome_trace
        if not enable:
            self.prof = None
            return

        levels = {
            0: torch_npu.profiler.ProfilerLevel.Level0,
            1: torch_npu.profiler.ProfilerLevel.Level1,
            2: torch_npu.profiler.ProfilerLevel.Level2,
        }
        if profiler_level not in levels:
            raise TypeError("profiler_level should be 0, 1 or 2")

        experimental_config = torch_npu.profiler._ExperimentalConfig(
            aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
            profiler_level=levels[profiler_level],
            l2_cache=False,
            data_simplification=False,
        )
        on_trace_ready = None
        if tensorboard_trace is not None:
            on_trace_ready = torch_npu.profiler.tensorboard_trace_handler(tensorboard_trace)

        self.prof = torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            on_trace_ready=on_trace_ready,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            experimental_config=experimental_config,
        )

    def __enter__(self):
        if self.prof is None:
            return None
        self.prof.__enter__()
        return self.prof

    def __exit__(self, exc_type, exc_value, traceback):
        if self.prof is None:
            return
        self.prof.__exit__(exc_type, exc_value, traceback)
        if self.chrome_trace is not None:
            self.prof.export_chrome_trace(self.chrome_trace)


def generate_seq_lengths(seq_len, num_batches):
    if seq_len == 0:
        return [0] * num_batches

    points = list(np.random.choice(range(1, seq_len), size=num_batches - 1, replace=False))
    points = sorted([0, seq_len] + points)
    return sorted(points[i + 1] - points[i] for i in range(num_batches))


def make_attention_mask(max_kv_len, kv_lens, query_lens):
    masks = []
    for kv_len, q_len in zip(kv_lens, query_lens):
        history_len = kv_len - q_len
        history = np.ones((q_len, history_len), dtype=np.float16)
        causal = np.tril(np.ones((q_len, max_kv_len - history_len), dtype=np.float16))
        mask = np.concatenate((history, causal), axis=1)
        mask[mask == 0] = -10000.0
        mask[mask == 1] = 0.0
        masks.append(mask)
    return torch.from_numpy(np.concatenate(masks, axis=0))


def convert_nd_to_nz(x):
    *prefix, m, n = x.shape
    m_pad = ((m + 15) // 16) * 16
    n_pad = ((n + 15) // 16) * 16

    if m != m_pad or n != n_pad:
        padded = np.zeros(tuple(prefix) + (m_pad, n_pad), dtype=x.dtype)
        padded[..., :m, :n] = x
        x = padded

    m1 = m_pad // 16
    n1 = n_pad // 16
    offset = len(prefix)
    axes = list(range(offset)) + [offset + 2, offset, offset + 1, offset + 3]
    return x.reshape(tuple(prefix) + (m1, 16, n1, 16)).transpose(axes)


def make_cache_nz(cache_nd):
    cache_np = cache_nd.detach().cpu().numpy().astype(np.float16)
    num_blocks, block_size, num_kv_heads, head_dim = cache_np.shape
    cache_nz = convert_nd_to_nz(
        cache_np.reshape(num_blocks, block_size, num_kv_heads * head_dim)
    )
    cache_nz = np.ascontiguousarray(cache_nz.reshape(num_blocks, -1, block_size, 16))
    return torch_npu.npu_format_cast(torch.from_numpy(cache_nz).npu(), 29)


def make_query_nz(query_nd, q_lens, num_kv_heads):
    num_tokens, num_heads, head_dim = query_nd.shape
    assert num_heads % num_kv_heads == 0
    assert sum(q_lens) == num_tokens
    assert head_dim % 16 == 0
    query_head_major = query_nd.permute(1, 0, 2).contiguous()
    return torch_npu.npu_format_cast(query_head_major, 29)


def make_mask_nz(max_kv_len, kv_lens, q_lens):
    mask_nd = make_attention_mask(max_kv_len, kv_lens, q_lens).numpy().astype(np.float16)
    num_tokens, max_seq_len = mask_nd.shape
    num_tokens_pad = ((num_tokens + 15) // 16) * 16
    max_seq_len_pad = ((max_seq_len + 15) // 16) * 16

    mask_pad = np.zeros((1, num_tokens_pad, max_seq_len_pad), dtype=np.float16)
    mask_pad[:, :num_tokens, :max_seq_len] = mask_nd
    mask_nz = convert_nd_to_nz(mask_pad)
    mask_nz = np.ascontiguousarray(mask_nz.reshape(1, -1, num_tokens_pad, 16))
    return torch_npu.npu_format_cast(torch.from_numpy(mask_nz).npu(), 29)


def compare_outputs(name, out, ref, ref_name="ref", eps=1e-1):
    if torch.isnan(out).any():
        print(f"\n========== {name} has NAN ==========")
        print(f"nan_indices: {torch.where(torch.isnan(out))}")
        return

    print(f"\n========== compare {name} vs {ref_name} ==========")
    out_np = out.to(torch.float32).cpu().numpy()
    ref_np = ref.to(torch.float32).cpu().numpy()
    abs_error, rel_error, top_indices = compute_diff(out_np, ref_np, idx_cnt=20)
    print_diff(out_np, ref_np, name, ref_name, abs_error, rel_error, top_indices, eps=eps)

    # P4 diagnostics: distinguish a whole 128-row slice failure from a tail or
    # isolated head/dimension error. Keep this in the existing test only.
    abs_diff = np.abs(out_np - ref_np)
    if abs_diff.size != 0 and np.max(abs_diff) > eps:
        reduce_axes = tuple(range(1, abs_diff.ndim))
        row_max = np.max(abs_diff, axis=reduce_axes)
        bad_rows = np.flatnonzero(row_max > eps)
        first_bad = tuple(np.argwhere(abs_diff > eps)[0])
        print(
            f"first error > {eps}: index={first_bad}, "
            f"{name}={out_np[first_bad]:.6f}, {ref_name}={ref_np[first_bad]:.6f}, "
            f"abs_diff={abs_diff[first_bad]:.6f}"
        )
        print(f"bad token rows: first={int(bad_rows[0])}, last={int(bad_rows[-1])}, count={bad_rows.size}")
        print("row max abs diff:")
        print([(int(row), float(row_max[row])) for row in bad_rows])

def create_inputs(
    num_batches,
    batch_seq_len,
    batch_kv_len,
    num_heads,
    num_kv_heads,
    head_size,
    page_size,
    seq_lengths_host,
    kv_lengths_host,
    dtype,
    device,
):
    if seq_lengths_host is None:
        seq_lengths_host = generate_seq_lengths(batch_seq_len, num_batches)
    if kv_lengths_host is None:
        history_kv_len = batch_kv_len - batch_seq_len
        history_lens = generate_seq_lengths(history_kv_len, num_batches)
        kv_lengths_host = [seq + hist for seq, hist in zip(seq_lengths_host, history_lens)]

    num_pages = sum((kv_len + page_size - 1) // page_size for kv_len in kv_lengths_host)
    q = torch.randn((batch_seq_len, num_heads, head_size), dtype=dtype, device=device)
    k_cache = torch.randn((num_pages, page_size, num_kv_heads, head_size), dtype=dtype, device=device)
    v_cache = torch.randn((num_pages, page_size, num_kv_heads, head_size), dtype=dtype, device=device)
    block_table = torch.stack(
        [torch.randperm(num_pages, dtype=torch.int32, device=device) for _ in range(num_batches)]
    )
    
    return q, k_cache, v_cache, block_table, seq_lengths_host, kv_lengths_host


def run_profiled(name, fn, profile_dir, profile_level, repeat, warmup, sync_each):
    for _ in range(warmup):
        out = fn()
    torch.npu.synchronize()

    with NpuProfile(
        chrome_trace=os.path.join(profile_dir, f"{name}.json"),
        tensorboard_trace=os.path.join(profile_dir, f"{name}_tb"),
        profiler_level=profile_level,
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
        enable=True,
    ):
        for _ in range(repeat):
            out = fn()
            if sync_each:
                torch.npu.synchronize()
        if not sync_each:
            torch.npu.synchronize()
    return out


def test_paged_attention(
    num_batches,
    batch_seq_len,
    batch_kv_len,
    num_heads,
    num_kv_heads,
    head_size,
    page_size=128,
    seq_lengths_host=None,
    kv_lengths_host=None,
):
    assert batch_kv_len >= batch_seq_len

    dtype = torch.float16
    device = "npu"
    softmax_scale = 1 / np.sqrt(head_size)

    q, k_cache, v_cache, block_table, seq_lengths_host, kv_lengths_host = create_inputs(
        num_batches,
        batch_seq_len,
        batch_kv_len,
        num_heads,
        num_kv_heads,
        head_size,
        page_size,
        seq_lengths_host,
        kv_lengths_host,
        dtype,
        device,
    )
    print(f"{seq_lengths_host=}")
    print(f"{kv_lengths_host=}")
    torch.npu.synchronize()

    k_cache_nz = make_cache_nz(k_cache)
    v_cache_nz = make_cache_nz(v_cache)
    q_nz = make_query_nz(q, seq_lengths_host, num_kv_heads)
    mask_col_size = max(
        ((kv_len + page_size - 1) // page_size) * page_size
        for kv_len in kv_lengths_host
    )
    mask_nz = make_mask_nz(mask_col_size, kv_lengths_host, seq_lengths_host)

    kv_lengths_mix = torch.tensor(kv_lengths_host, dtype=torch.int64, device=device)
    seq_lengths_mix = torch.tensor(seq_lengths_host, dtype=torch.int64, device=device)
    kv_lengths_atb = torch.tensor(kv_lengths_host, dtype=torch.int32, device=device)
    seq_lengths_atb = torch.tensor(seq_lengths_host, dtype=torch.int32)

    def run_mix_v3():
        return torch.ops.npu.paged_attention_mix_v3(
            q_nz,
            k_cache_nz,
            v_cache_nz,
            mask_nz,
            block_table,
            kv_lengths_mix,
            seq_lengths_mix,
            None,
            float(softmax_scale),
            False,
        )

    out_splitfuse = None

    def run_splitfuse():
        nonlocal out_splitfuse
        if out_splitfuse is None:
            out_splitfuse = torch.empty_like(out_mix_v3)
        torch_npu._npu_paged_attention_splitfuse(
            query=q,
            key_cache=k_cache_nz,
            value_cache=v_cache_nz,
            mask=mask_nz,
            block_table=block_table,
            context_lens=kv_lengths_atb,
            seq_len=seq_lengths_atb,
            scale_value=softmax_scale,
            num_kv_heads=num_kv_heads,
            num_heads=num_heads,
            out=out_splitfuse,
        )
        return out_splitfuse

    target, profile_mix_v3, profile_atb = profile_targets()
    profile_enabled = profile_mix_v3 or profile_atb
    profile_level = int(os.getenv("PROFILE_LEVEL", "1"))
    profile_repeat = int(os.getenv("PROFILE_REPEAT", "10"))
    profile_warmup = int(os.getenv("PROFILE_WARMUP", "3"))
    profile_dir = os.getenv("PROFILE_DIR", "profiler_attention")
    if profile_enabled:
        os.makedirs(profile_dir, exist_ok=True)
        print(
            f"\nPROFILE enabled: target={target or 'env flags'}, "
            f"repeat={profile_repeat}, warmup={profile_warmup}, "
            f"level={profile_level}, dir={profile_dir}"
        )

    if profile_mix_v3:
        out_mix_v3 = run_profiled(
            "mix_v3",
            run_mix_v3,
            profile_dir,
            profile_level,
            profile_repeat,
            profile_warmup,
            sync_each=True,
        )
    else:
        out_mix_v3 = run_mix_v3()
        torch.npu.synchronize()

    if profile_atb:
        out_splitfuse = run_profiled(
            "atb_splitfuse",
            run_splitfuse,
            profile_dir,
            profile_level,
            profile_repeat,
            profile_warmup,
            sync_each=False,
        )
    else:
        out_splitfuse = run_splitfuse()
        torch.npu.synchronize()

    compare_outputs("mix_v3", out_mix_v3, out_splitfuse, ref_name="splitfuse")


def main():
    main_repeat = 1 if is_profile_enabled() else int(os.getenv("TEST_REPEAT", "1"))
    for _ in range(main_repeat):
        # P4 exact 128-row slices with one visible KV tile.
        # test_paged_attention(2, 256, 512, 32, 4, 128, 128, [128, 128], [128, 128])
        # P4 main multi-KV-tile regression: 64 and 128+64 token slices.
        test_paged_attention(2, 256, 512, 32, 4, 128, 128, [64, 192], [80, 432])
        # Non-16-aligned tails: 63 and 128+65 token slices.
        # test_paged_attention(2, 256, 512, 32, 4, 128, 128, [63, 193], [80, 432])
        # P4 boundary coverage, including one-row tails inside aligned total-Q input.
        # test_paged_attention(2, 256, 512, 32, 4, 128, 128, [127, 129], [160, 432])
        # test_paged_attention(2, 256, 512, 32, 4, 128, 128, [129, 127], [432, 160])
        # test_paged_attention(2, 256, 512, 32, 4, 128, 128, [1, 255], [80, 432])
        # test_paged_attention(2, 256, 512, 32, 4, 128, 128, [17, 239], [80, 432])
        # Decode regressions.  KV lengths 1/128/512 separate the one-row
        # Q/output path from single-tile and multi-tile online-softmax behavior.
        # The second positional argument is total Q tokens and must equal
        # sum(seq_lengths_host), not the per-request qLen.
        # test_paged_attention(1, 1, 1, 32, 4, 128, 128, [1], [1])
        # test_paged_attention(1, 1, 128, 32, 4, 128, 128, [1], [128])
        # test_paged_attention(1, 1, 512, 32, 4, 128, 128, [1], [512])
        # test_paged_attention(2, 2, 512, 32, 4, 128, 128, [1, 1], [80, 432])
        # test_paged_attention(2, 2560, 5120, 96, 8, 128, 128)


if __name__ == "__main__":
    main()
