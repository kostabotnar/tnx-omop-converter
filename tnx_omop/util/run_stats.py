"""Wall time and peak memory per pipeline stage, with the standard library only.

The operating system reports the peak of a whole process and cannot reset it per
stage on Windows. A daemon thread therefore samples the current memory of the process
about every 0.2 s and keeps the maximum for every active stage. A stage shorter than
the interval is still sampled when it starts and when it ends. The peak of a stage is
a sampled value and can miss a spike between two samples.

Counters per platform (`MemoryReader.counter`):

- Windows: current = `PrivateUsage`, process peak = `PeakPagefileUsage` (both of
  `PROCESS_MEMORY_COUNTERS_EX`). The working set is not used because it undercounts
  the memory that Polars allocates and the system pages out.
- Linux: current = resident set size from `/proc/self/statm`, process peak =
  `ru_maxrss` (kilobytes).
- macOS: process peak = `ru_maxrss` (bytes). There is no cheap current value, so the
  stage peaks are unknown (None) and only the process peak is recorded.
"""

from __future__ import annotations

import ctypes
import logging
import os
import sys
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from types import TracebackType
from typing import Protocol

logger = logging.getLogger(__name__)

SAMPLE_INTERVAL_SECONDS = 0.2
UNKNOWN_COUNTER = "unknown"


class MemoryReader(Protocol):
    """Reads the memory of the current process in bytes; None when not available."""

    counter: str

    def current(self) -> int | None:
        """Memory in use right now."""

    def process_peak(self) -> int | None:
        """Highest memory of the process since it started, from the OS."""


class _ProcessMemoryCountersEx(ctypes.Structure):
    """PROCESS_MEMORY_COUNTERS_EX of the Windows API."""

    _fields_ = (
        ("cb", ctypes.c_ulong),
        ("PageFaultCount", ctypes.c_ulong),
        ("PeakWorkingSetSize", ctypes.c_size_t),
        ("WorkingSetSize", ctypes.c_size_t),
        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPagedPoolUsage", ctypes.c_size_t),
        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
        ("PagefileUsage", ctypes.c_size_t),
        ("PeakPagefileUsage", ctypes.c_size_t),
        ("PrivateUsage", ctypes.c_size_t),
    )


class WindowsMemoryReader:
    """Private bytes (current) and peak pagefile usage (process peak)."""

    counter = "PrivateUsage, PeakPagefileUsage"

    def __init__(self) -> None:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self._get_current_process = kernel32.GetCurrentProcess
        self._get_current_process.restype = ctypes.c_void_p
        self._get_info = kernel32.K32GetProcessMemoryInfo
        self._get_info.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCountersEx),
            ctypes.c_ulong,
        )
        self._get_info.restype = ctypes.c_int

    def _counters(self) -> _ProcessMemoryCountersEx | None:
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        ok = self._get_info(
            self._get_current_process(), ctypes.byref(counters), counters.cb
        )
        return counters if ok else None

    def current(self) -> int | None:
        counters = self._counters()
        return counters.PrivateUsage if counters else None

    def process_peak(self) -> int | None:
        counters = self._counters()
        return counters.PeakPagefileUsage if counters else None


class LinuxMemoryReader:
    """Resident set size (current) and `ru_maxrss` in kilobytes (process peak)."""

    counter = "VmRSS, ru_maxrss"

    def __init__(self) -> None:
        self._page_size = os.sysconf("SC_PAGE_SIZE")

    def current(self) -> int | None:
        try:
            with open("/proc/self/statm", encoding="ascii") as f:
                return int(f.read().split()[1]) * self._page_size
        except (OSError, ValueError, IndexError):
            return None

    def process_peak(self) -> int | None:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024


class MacMemoryReader:
    """`ru_maxrss` in bytes (process peak only)."""

    counter = "ru_maxrss"

    def current(self) -> int | None:
        return None

    def process_peak(self) -> int | None:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


class NullMemoryReader:
    """Reader for a platform without a known counter."""

    counter = UNKNOWN_COUNTER

    def current(self) -> int | None:
        return None

    def process_peak(self) -> int | None:
        return None


def default_reader() -> MemoryReader:
    """Return the memory reader of the current platform."""
    try:
        if sys.platform == "win32":
            return WindowsMemoryReader()
        if sys.platform == "darwin":
            return MacMemoryReader()
        if sys.platform.startswith("linux"):
            return LinuxMemoryReader()
    except (OSError, AttributeError, ImportError):
        logger.debug("No memory counter available", exc_info=True)
    return NullMemoryReader()


@dataclass
class StageStats:
    """Measurements of one stage.

    Attributes:
        name: Stage name.
        seconds: Wall time; it adds up when a stage name is used more than once.
        peak_memory_bytes: Highest sampled memory during the stage, None when the
            platform has no current memory counter.
    """

    name: str
    seconds: float = 0.0
    peak_memory_bytes: int | None = None


class RunStats:
    """Collect time and peak memory per stage while the sampling thread runs.

    Use as a context manager around the whole run and `stage()` around each stage:

        with RunStats() as stats:
            with stats.stage("lookup"):
                ...
    """

    def __init__(
        self,
        reader: MemoryReader | None = None,
        interval: float = SAMPLE_INTERVAL_SECONDS,
    ) -> None:
        self._reader = reader if reader is not None else default_reader()
        self._interval = interval
        self._lock = threading.Lock()
        self._active: list[StageStats] = []
        self._stages: dict[str, StageStats] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.process_peak_bytes: int | None = None

    @property
    def counter(self) -> str:
        """Name of the memory counters in use."""
        return self._reader.counter

    @property
    def stages(self) -> dict[str, StageStats]:
        """Finished and running stages by name, in the order they started."""
        with self._lock:
            return dict(self._stages)

    def __enter__(self) -> RunStats:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._sample_loop, name="run-stats-sampler", daemon=True
        )
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.process_peak_bytes = self._reader.process_peak()

    def _sample(self) -> None:
        current = self._reader.current()
        if current is None:
            return
        with self._lock:
            for stats in self._active:
                if stats.peak_memory_bytes is None or current > stats.peak_memory_bytes:
                    stats.peak_memory_bytes = current

    def _sample_loop(self) -> None:
        while not self._stop.wait(self._interval):
            self._sample()

    @contextmanager
    def stage(self, name: str) -> Iterator[StageStats]:
        """Measure the wall time and peak memory of the enclosed block.

        The time and the memory peak are recorded when the block raises as well.
        """
        with self._lock:
            stats = self._stages.setdefault(name, StageStats(name))
            self._active.append(stats)
        self._sample()
        start = time.perf_counter()
        try:
            yield stats
        finally:
            elapsed = time.perf_counter() - start
            self._sample()
            with self._lock:
                self._active.remove(stats)
                stats.seconds += elapsed
            logger.debug(
                "Stage %s: %.1f s, peak %s bytes",
                name,
                elapsed,
                stats.peak_memory_bytes,
            )
