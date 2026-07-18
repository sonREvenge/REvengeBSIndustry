import threading
import time

_last = time.time()
_lock = threading.Lock()


def bump():
    global _last
    with _lock:
        _last = time.time()


def last_beat() -> float:
    with _lock:
        return _last
