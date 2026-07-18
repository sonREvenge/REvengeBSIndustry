import os
import threading
from utils import load_toml_as_dict, save_dict_as_toml, resolve_cfg_path
from brawl_industry_logger import get_logger

log = get_logger("brawl_industry.trophy")


class TrophyObserver:

    def __init__(self):
        self._lock = threading.Lock()
        self.history_file  = resolve_cfg_path("./cfg/match_history.toml")
        self.match_history = self._load_history()
        self._uncommitted_time: float = 0.0

    def _load_history(self) -> dict:
        loaded = load_toml_as_dict(self.history_file) if os.path.exists(self.history_file) else {}
        loaded.setdefault("total", {"defeat": 0, "victory": 0, "draw": 0, "time_played": 0.0})
        loaded["total"].setdefault("time_played", 0.0)
        return loaded

    def save_history(self):
        save_dict_as_toml(self.match_history, self.history_file)

    def reset_history(self):
        with self._lock:
            self.match_history     = {"total": {"defeat": 0, "victory": 0, "draw": 0, "time_played": 0.0}}
            self._uncommitted_time = 0.0
            self.save_history()

    def add_time_played(self, seconds: float):
        with self._lock:
            self.match_history["total"]["time_played"] += seconds
            self._uncommitted_time += seconds
            if self._uncommitted_time >= 30:
                self.save_history()
                self._uncommitted_time = 0.0

    def record_result(self, game_result: str):
        if game_result not in ("victory", "defeat", "draw"):
            log.warning(f"Unknown game result: {game_result}")
            return
        with self._lock:
            self.match_history["total"][game_result] += 1
            self.save_history()
            total = self.match_history["total"]
        log.info(f"Match result: {game_result} | W:{total['victory']} L:{total['defeat']} D:{total['draw']}")

    def get_session_stats(self) -> dict:
        with self._lock:
            total   = self.match_history.get("total", {})
            wins    = total.get("victory", 0)
            losses  = total.get("defeat",  0)
            draws   = total.get("draw",    0)
            games   = wins + losses + draws
            winrate = (wins / games * 100) if games > 0 else 0.0
            return {
                "wins":        wins,
                "losses":      losses,
                "draws":       draws,
                "games":       games,
                "winrate":     winrate,
                "time_played": total.get("time_played", 0.0),
            }
