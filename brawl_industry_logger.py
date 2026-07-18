import time as _time
from collections import deque


_log_queue: deque = deque(maxlen=5000)


class _Logger:
    __slots__ = ("name",)

    def __init__(self, name: str):
        self.name = name

    @staticmethod
    def _ts() -> str:
        return _time.strftime("%H:%M:%S")

    def _emit(self, level: str, msg):
        line = f"[{self._ts()}] [{level}] {msg}"
        print(line)
        _log_queue.append(line)

    def info(self, msg):    self._emit("INFO", msg)
    def warning(self, msg): self._emit("WARN", msg)
    def error(self, msg):   self._emit("ERR",  msg)
    def debug(self, msg):   pass


_loggers: dict = {}


def get_logger(name: str = "brawl_industry") -> _Logger:
    if name not in _loggers:
        _loggers[name] = _Logger(name)
    return _loggers[name]


def pop_logs(max_lines: int = 20) -> list[str]:
    out = []
    while _log_queue and len(out) < max_lines:
        out.append(_log_queue.popleft())
    return out
