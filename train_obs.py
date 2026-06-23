"""
train_obs.py — observability helpers shared by the FSDP SFT/DPO scripts.

Provides:
  - StepLogger: rich console line + append-only JSONL (rank0 only), parseable for plots.
  - nvtx_range(): context manager that emits NVTX ranges so nsys timelines are
    annotated with named phases (forward / backward / optim / ref_forward).
  - reduce_mean(): all-reduce a python float across ranks for global metrics.

NVTX ranges are nearly free when nsys is NOT attached, so you can leave them in.
"""

import json
import os
import time
from contextlib import contextmanager

import torch
import torch.distributed as dist

try:
    import torch.cuda.nvtx as nvtx
    _HAS_NVTX = True
except Exception:
    _HAS_NVTX = False


@contextmanager
def nvtx_range(name: str):
    """Annotate a code region on the nsys timeline. No-op if NVTX unavailable."""
    if _HAS_NVTX:
        nvtx.range_push(name)
    try:
        yield
    finally:
        if _HAS_NVTX:
            nvtx.range_pop()


def reduce_mean(value: float) -> float:
    """Average a scalar across all ranks (for global throughput/metrics)."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    t = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return (t / dist.get_world_size()).item()


def reduce_sum(value: float) -> float:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return value
    t = torch.tensor([value], device="cuda", dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t.item()


class StepLogger:
    """Console + JSONL logger. Only rank0 writes. One JSON object per line."""

    def __init__(self, jsonl_path: str, run_name: str, enabled: bool):
        self.enabled = enabled
        self.run_name = run_name
        self.path = jsonl_path
        if self.enabled:
            os.makedirs(os.path.dirname(jsonl_path) or ".", exist_ok=True)
            # truncate at start of run
            self._f = open(jsonl_path, "w")
        else:
            self._f = None

    def log(self, **fields):
        """fields: arbitrary scalars. Always includes wall_time + run_name."""
        if not self.enabled:
            return
        rec = {"run": self.run_name, "wall_time": time.time(), **fields}
        self._f.write(json.dumps(rec) + "\n")
        self._f.flush()
        # human-readable console line
        parts = []
        for k, v in fields.items():
            if isinstance(v, float):
                if abs(v) >= 1000 or (v != 0 and abs(v) < 1e-3):
                    parts.append(f"{k}={v:.3e}")
                else:
                    parts.append(f"{k}={v:.4f}")
            else:
                parts.append(f"{k}={v}")
        print(f"[{self.run_name}] " + " ".join(parts), flush=True)

    def close(self):
        if self._f:
            self._f.close()


class MeanAccumulator:
    """Accumulate scalars across micro-batches, then reduce-average across ranks.

    Usage per optimizer step:
        acc = MeanAccumulator()
        for each micro-batch: acc.add(loss_value)   # the UN-divided loss
        ...
        step_loss = acc.global_mean()               # mean over micro-batches AND ranks
        acc.reset()

    global_mean() sums (value, count) locally, all-reduces both, divides. This gives
    the true batch mean even if ranks saw a different number of micro-batches.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._sum = 0.0
        self._n = 0

    def add(self, value: float, weight: float = 1.0):
        self._sum += float(value) * weight
        self._n += weight

    def global_mean(self) -> float:
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return self._sum / self._n if self._n else 0.0
        t = torch.tensor([self._sum, self._n], device="cuda", dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        s, n = t[0].item(), t[1].item()
        return s / n if n else 0.0