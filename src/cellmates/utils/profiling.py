import glob
import os
import time
import threading
from collections import defaultdict


class HMMProfiler:
    """Global profiler for HMM-related timings.

    Supports low-level timings (forward-backward / viterbi) and
    high-level timings (one EM iteration).

    In multiprocessing runs, each process can write into its own worker
    log file (suffix `.worker_<pid>.log`). The main process can then merge
    those logs into the main profiling log.
    """

    def __init__(self):
        self.enabled = False
        self.log_path = None
        self._worker_log_path = None
        self._lock = threading.Lock()
        self._stats = defaultdict(lambda: {"count": 0, "total": 0.0})

    def configure(self, enabled: bool, log_path: str | None = None):
        self.enabled = bool(enabled)
        self.log_path = log_path
        self._worker_log_path = None
        self._stats = defaultdict(lambda: {"count": 0, "total": 0.0})
        if self.enabled and self.log_path is not None:
            log_dir = os.path.dirname(self.log_path) or "."
            os.makedirs(log_dir, exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("\n=== HMM profiling session start ===\n")

    def configure_worker(self):
        """Enable per-worker logging for current process.

        Safe to call only after `configure(enabled=True, log_path=...)`.
        """
        if not self.enabled or self.log_path is None:
            return
        base, ext = os.path.splitext(self.log_path)
        pid = os.getpid()
        self._worker_log_path = f"{base}.worker_{pid}{ext or '.log'}"
        with self._lock:
            with open(self._worker_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n=== HMM profiling worker start pid={pid} ===\n")

    def _append(self, line: str):
        if not self.enabled or self.log_path is None:
            return
        path = self._worker_log_path or self.log_path
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
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

    def merge_worker_logs(self):
        """Merge all worker logs into main log and append merged summary.

        This method:
        1) scans files matching `<base>.worker_*<ext>`
        2) appends all worker [timing] lines to the main log
        3) computes and appends an aggregated [timing-summary]
        """
        if not self.enabled or self.log_path is None:
            return

        base, ext = os.path.splitext(self.log_path)
        pattern = f"{base}.worker_*{ext or '.log'}"
        worker_logs = sorted(glob.glob(pattern))
        if not worker_logs:
            return

        merged_stats = defaultdict(lambda: {"count": 0, "total": 0.0})

        def _parse_timing_line(line: str):
            # format:
            # [timing] <category>: <seconds>s (avg=..., n=...)
            if not line.startswith("[timing] "):
                return None
            try:
                body = line[len("[timing] "):]
                category, rest = body.split(": ", 1)
                seconds_txt = rest.split("s", 1)[0]
                return category, float(seconds_txt)
            except Exception:
                return None

        with self._lock:
            with open(self.log_path, "a", encoding="utf-8") as out:
                out.write("[timing-merge] begin\n")
                for wpath in worker_logs:
                    out.write(f"[timing-merge] source={wpath}\n")
                    with open(wpath, "r", encoding="utf-8") as wf:
                        for raw in wf:
                            line = raw.rstrip("\n")
                            if line.startswith("[timing] "):
                                out.write(line + "\n")
                                parsed = _parse_timing_line(line)
                                if parsed is not None:
                                    category, seconds = parsed
                                    merged_stats[category]["count"] += 1
                                    merged_stats[category]["total"] += seconds

                out.write("[timing-summary] begin\n")
                for category in sorted(merged_stats.keys()):
                    count = merged_stats[category]["count"]
                    total = merged_stats[category]["total"]
                    avg = total / count if count > 0 else 0.0
                    out.write(
                        f"[timing-summary] {category}: avg={avg:.6f}s, total={total:.6f}s, n={count}\n"
                    )
                out.write("[timing-summary] end\n")
                out.write("[timing-merge] end\n")


hmm_profiler = HMMProfiler()


def timed_call(category: str, fn, *args, meta: dict | None = None, **kwargs):
    if not hmm_profiler.enabled:
        return fn(*args, **kwargs)
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    dt = time.perf_counter() - t0
    hmm_profiler.record(category, dt, meta=meta)
    return out
