import os
import time
import threading
from collections import defaultdict


class HMMProfiler:
    """Global profiler for HMM-related timings.

    Supports low-level timings (forward-backward / viterbi) and
    high-level timings (one EM iteration).
    """

    def __init__(self):
        self.enabled = False
        self.log_path = None
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: {"count": 0, "total": 0.0})

    def configure(self, enabled: bool, log_path: str | None = None):
        self.enabled = bool(enabled)
        self.log_path = log_path
        if self.enabled and self.log_path is not None:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n=== HMM profiling session start ===\n")

    def _append(self, line: str):
        if not self.enabled or self.log_path is None:
            return
        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def record(self, category: str, seconds: float, meta: dict | None = None):
        if not self.enabled:
            return
        key = category
        with self._lock:
            self._stats[key]["count"] += 1
            self._stats[key]["total"] += float(seconds)
            count = self._stats[key]["count"]
            avg = self._stats[key]["total"] / count

        meta_txt = ""
        if meta:
            parts = [f"{k}={v}" for k, v in meta.items()]
            meta_txt = " | " + ", ".join(parts)
        self._append(f"[timing] {category}: {seconds:.6f}s (avg={avg:.6f}s, n={count}){meta_txt}")

    def summary(self):
        if not self.enabled:
            return
        self._append("[timing-summary] begin")
        with self._lock:
            items = list(self._stats.items())
        for category, val in items:
            count = val["count"]
            total = val["total"]
            avg = total / count if count > 0 else 0.0
            self._append(f"[timing-summary] {category}: avg={avg:.6f}s, total={total:.6f}s, n={count}")
        self._append("[timing-summary] end")


hmm_profiler = HMMProfiler()


def timed_call(category: str, fn, *args, meta: dict | None = None, **kwargs):
    if not hmm_profiler.enabled:
        return fn(*args, **kwargs)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    hmm_profiler.record(category, dt, meta=meta)
    return out
