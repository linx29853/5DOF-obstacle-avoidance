"""本地评测的内存保护；在低内存时中断并保留已完成记录。"""

from pathlib import Path
from time import monotonic


class ResourcePressure(RuntimeError):
    pass


def memory_snapshot():
    """Linux 使用可用内存与当前进程 RSS，单位 byte。"""
    mem = Path("/proc/meminfo")
    status = Path("/proc/self/status")
    if not mem.exists() or not status.exists():
        raise ResourcePressure("内存保护需要 Linux /proc 内存数据")
    values = {}
    for path in [mem, status]:
        for line in path.read_text().splitlines():
            parts = line.split()
            if parts and parts[0] in ("MemAvailable:", "VmRSS:"):
                values[parts[0][:-1]] = int(parts[1]) * 1024
    if len(values) != 2:
        raise ResourcePressure("无法读取内存状态，停止评测")
    return values


class MemoryGuard:
    def __init__(self, min_available=3 * 1024**3, max_rss=1536 * 1024**2, interval=0.5):
        self.min_available = min_available
        self.max_rss = max_rss
        self.interval = interval
        self.last = -float("inf")
        self.peak_rss = 0

    def check(self, force=False):
        now = monotonic()
        if not force and now - self.last < self.interval:
            return
        self.last = now
        snapshot = memory_snapshot()
        self.peak_rss = max(self.peak_rss, snapshot["VmRSS"])
        if snapshot["MemAvailable"] < self.min_available:
            raise ResourcePressure("系统可用内存低于 3 GiB，评测已停止；已完成记录保留")
        if snapshot["VmRSS"] > self.max_rss:
            raise ResourcePressure(
                "评测进程内存超过 1.5 GiB，评测已停止；已完成记录保留"
            )


_ACTIVE = None


def enable_guard():
    global _ACTIVE
    _ACTIVE = MemoryGuard()
    _ACTIVE.check(force=True)
    return _ACTIVE


def check_resources():
    if _ACTIVE is not None:
        _ACTIVE.check()
