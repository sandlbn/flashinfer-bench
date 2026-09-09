"""Substitute the tuned paged-decode kernel for vLLM's unified attention, in place.

Installed by setting FIB_PATCH_ATTENTION=1 before vLLM imports. It replaces
`unified_attention` with a wrapper that uses the tuned kernel only for the case it was
tuned and validated for -- a pure decode step, no sliding window, no softcap, no
quantisation descales, head_dim a power of two -- and calls the original for everything
else. Anything outside that envelope is not a slow path here, it is an unvalidated one, so
it is not taken.

The wrapper counts what it handled and what it declined, and prints the tally at exit. A
throughput number without that tally cannot be read: zero substitutions and a healthy
speedup means something else moved.
"""

from __future__ import annotations

import atexit
import os
import sys

_STATS = {"tuned": 0, "fell_back": 0}
_REASONS: dict[str, int] = {}


def _decline(reason: str):
    _STATS["fell_back"] += 1
    _REASONS[reason] = _REASONS.get(reason, 0) + 1


def install() -> bool:
    if os.environ.get("FIB_PATCH_ATTENTION", "").lower() not in ("1", "true", "yes", "on"):
        return False
    try:
        from vllm.v1.attention.ops import triton_unified_attention as tua
    except Exception as exc:  # pragma: no cover - vLLM absent
        print(f"[fib-attn] vLLM not importable: {exc}", flush=True)
        return False

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from paged_decode_attention import Model  # the finalized trial

    original = tua.unified_attention
    model = Model()

    def patched(q, k, v, out, cu_seqlens_q, max_seqlen_q, seqused_k, max_seqlen_k,
                softmax_scale, causal, window_size, block_table, softcap,
                q_descale, k_descale, v_descale, **kwargs):
        # Name the specific reason. Bucketing every rejection as "unsupported" made a
        # 0/14672 substitution rate undiagnosable -- the tally said the kernel never ran and
        # could not say which condition to relax.
        reason = None
        if max_seqlen_q != 1:
            reason = "prefill"
        elif softcap:
            reason = "softcap"
        elif tuple(window_size) != (-1, -1):
            reason = f"sliding-window{tuple(window_size)}"
        elif not (q_descale is None and k_descale is None and v_descale is None):
            reason = "quantised-descale"
        elif q.shape[-1] not in (64, 128, 256):
            reason = f"head_dim={q.shape[-1]}"
        elif q.shape[1] % k.shape[2] != 0:
            reason = f"heads {q.shape[1]}/{k.shape[2]}"
        elif kwargs.get("sinks") is not None:
            reason = "sinks"
        elif q.shape[0] != seqused_k.shape[0]:
            reason = f"rows {q.shape[0]} != seqs {seqused_k.shape[0]}"
        if reason is not None:
            _decline(reason)
            return original(q=q, k=k, v=v, out=out, cu_seqlens_q=cu_seqlens_q,
                            max_seqlen_q=max_seqlen_q, seqused_k=seqused_k,
                            max_seqlen_k=max_seqlen_k, softmax_scale=softmax_scale,
                            causal=causal, window_size=window_size,
                            block_table=block_table, softcap=softcap,
                            q_descale=q_descale, k_descale=k_descale,
                            v_descale=v_descale, **kwargs)
        model.scale = softmax_scale
        out.copy_(model(q, k, v, cu_seqlens_q, seqused_k, block_table))
        _STATS["tuned"] += 1
        return None

    tua.unified_attention = patched
    print("[fib-attn] tuned paged-decode kernel installed", flush=True)

    @atexit.register
    def _report() -> None:
        total = _STATS["tuned"] + _STATS["fell_back"]
        if not total:
            print("[fib-attn] unified_attention was never called", flush=True)
            return
        pct = 100.0 * _STATS["tuned"] / total
        print(f"[fib-attn] {_STATS['tuned']}/{total} call(s) used the tuned kernel "
              f"({pct:.1f}%); declined: {_REASONS or 'none'}", flush=True)

    return True


install()
