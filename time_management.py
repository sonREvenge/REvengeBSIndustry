import time
from utils import load_toml_as_dict


class TimeManagement:
    def __init__(self):
        self.thresholds = load_toml_as_dict("cfg/time_thresholds.toml")
        self.states = {key: time.time() for key in self.thresholds}

    def check_time(self, check_type: str) -> bool:
        now = time.time()
        if now - self.states[check_type] >= self.thresholds[check_type]:
            self.states[check_type] = now
            return True
        return False

    def state_check(self):        return self.check_time("state_check")
    def no_detections_check(self): return self.check_time("no_detections")
    def idle_check(self):          return self.check_time("idle")
