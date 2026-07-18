import time
from utils import count_hsv_pixels
from brawl_industry_logger import get_logger

log = get_logger("brawl_industry.lobby")

_BLUE_SCREEN_HSV_LOW   = (100, 150, 150)
_BLUE_SCREEN_HSV_HIGH  = (120, 255, 255)
_BLUE_SCREEN_THRESHOLD = 0.60
_BLUE_SCREEN_DURATION  = 15.0

_IDLE_HSV_LOW   = (0, 0, 55)
_IDLE_HSV_HIGH  = (10, 15, 77)
_IDLE_THRESHOLD = 1000


class LobbyAutomation:

    def __init__(self, window_controller):
        self.window_controller = window_controller
        self._blue_screen_since: float | None = None

    def _is_blue_screen(self, frame) -> bool:
        h, w = frame.shape[:2]
        total = h * w
        if total == 0:
            return False
        return count_hsv_pixels(frame, _BLUE_SCREEN_HSV_LOW, _BLUE_SCREEN_HSV_HIGH) / total >= _BLUE_SCREEN_THRESHOLD

    def _do_restart(self):
        try:
            self.window_controller.restart_brawl_stars()
            self.window_controller.click(535, 615)
        except (ConnectionError, OSError):
            raise
        except Exception as e:
            log.warning(f"restart_brawl_stars failed: {e}")

    def check_for_idle(self, frame):
        if self._is_blue_screen(frame):
            if self._blue_screen_since is None:
                self._blue_screen_since = time.time()
            elif time.time() - self._blue_screen_since >= _BLUE_SCREEN_DURATION:
                log.warning("Blue crash screen confirmed, restarting Brawl Stars")
                self._blue_screen_since = None
                self._do_restart()
            return

        self._blue_screen_since = None

        x0, x1 = 400, 1500
        y0, y1 = 380, 700
        if count_hsv_pixels(frame[y0:y1, x0:x1], _IDLE_HSV_LOW, _IDLE_HSV_HIGH) > _IDLE_THRESHOLD:
            self._do_restart()
