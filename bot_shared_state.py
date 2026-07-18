import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Deque, Optional


@dataclass(eq=False)
class BotState:
    running:             bool            = False
    stuck_alert:         bool            = False
    pending_reload:      bool            = False
    current_ips:         float           = 0.0
    session_start_time:  Optional[float] = None
    current_brawler:     Optional[str]   = None

    session_stats_func:  Optional[Callable] = None
    reset_stats_func:    Optional[Callable] = None
    get_screenshot:      Optional[Callable] = None
    restart_func:        Optional[Callable] = None

    _pending_notifications: Deque[str] = field(
        default_factory=lambda: deque(maxlen=200),
        repr=False,
    )
    _last_notif_wins:  int = field(default=0, repr=False)
    _last_notif_games: int = field(default=0, repr=False)

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def is_running(self) -> bool:
        with self._lock: return self.running

    def set_running(self, state: bool):
        with self._lock: self.running = state

    def set_stuck_alert(self, state: bool):
        with self._lock: self.stuck_alert = state

    def get_stuck_alert(self) -> bool:
        with self._lock: return self.stuck_alert

    def take_stuck_alert(self) -> bool:
        with self._lock:
            val, self.stuck_alert = self.stuck_alert, False
            return val

    def set_pending_reload(self):
        with self._lock: self.pending_reload = True

    def take_pending_reload(self) -> bool:
        with self._lock:
            val = self.pending_reload
            self.pending_reload = False
            return val

    def update_ips(self, ips: float):
        with self._lock: self.current_ips = ips

    def get_ips(self) -> float:
        with self._lock: return self.current_ips

    def get_session_start_time(self) -> Optional[float]:
        with self._lock: return self.session_start_time

    def set_session_start_time(self, t: Optional[float]):
        with self._lock: self.session_start_time = t

    def get_current_brawler(self) -> Optional[str]:
        with self._lock: return self.current_brawler

    def set_current_brawler(self, brawler: str):
        with self._lock: self.current_brawler = brawler

    def get_last_notif_wins(self) -> int:
        with self._lock: return self._last_notif_wins

    def set_last_notif_wins(self, val: int):
        with self._lock: self._last_notif_wins = val

    def get_last_notif_games(self) -> int:
        with self._lock: return self._last_notif_games

    def set_last_notif_games(self, val: int):
        with self._lock: self._last_notif_games = val

    def push_notification(self, msg: str):
        with self._lock: self._pending_notifications.append(msg)

    def pop_notification(self) -> Optional[str]:
        with self._lock:
            return self._pending_notifications.popleft() if self._pending_notifications else None
