import time
import cv2

from state_finder import get_state
from trophy_observer import TrophyObserver
from utils import find_template_center, load_toml_as_dict
from brawl_industry_logger import get_logger
from _paths import img_state
import heartbeat

log = get_logger("brawl_industry.stage")

END_SCREEN_TIMEOUT = 25
PLAY_AGAIN_TIMEOUT = 25


def load_image(image_path: str):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found: '{image_path}'. Run the bot from the project root.")
    return image


class StageManager:

    def __init__(self, lobby_automator, window_controller):
        self.Lobby_automation = lobby_automator
        self.close_popup_icon = None
        self.Trophy_observer  = TrophyObserver()
        self.long_press_star_drop = load_toml_as_dict("./cfg/general_config.toml").get("long_press_star_drop", "no")
        self.play_again_on_win    = load_toml_as_dict("./cfg/bot_config.toml").get("play_again_on_win", "no") == "yes"
        self.window_controller    = window_controller
        self.states = {
            "shop":               self.quit_shop,
            "brawler_selection":  self.quit_shop,
            "popup":              self.close_pop_up,
            "match":              lambda: 0,
            "match_making":       lambda: 0,
            "end":                self.end_game,
            "end_victory":        self.end_game,
            "end_defeat":         self.end_game,
            "end_draw":           self.end_game,
            "lobby":              self.start_game,
            "star_drop":          self.click_star_drop,
            "trophy_reward":      lambda: self.window_controller.press_key("Q"),
            "prestige_milestone": lambda: self.window_controller.press_key("Q"),
            "nano_noodles":       self.click_nano_noodles,
        }

    def start_game(self):
        log.info("Lobby detected, starting a match")
        self.window_controller.keys_up()
        self.window_controller.press_key("Q")

    def click_star_drop(self):
        delay = 10 if self.long_press_star_drop == "yes" else 0.005
        self.window_controller.press_key("Q", delay)

    def click_nano_noodles(self):
        wc = self.window_controller
        cx, cy = 960, 740
        for offset in (0, 330, -330):
            wc.click(cx + offset, cy, already_include_ratio=False)
            time.sleep(0.1)

    def _safe_screenshot(self):
        try:
            return self.window_controller.screenshot()
        except (ConnectionError, OSError):
            return None

    def _scrcpy_disconnected(self) -> bool:
        return self.window_controller._scrcpy_disconnected.is_set()

    def end_game(self):
        screenshot = self._safe_screenshot()
        if screenshot is None:
            return

        current_state = get_state(screenshot)
        found_game_result = (
            current_state.split("_", 1)[1] if current_state.startswith("end_") else None
        )
        if found_game_result:
            self.Trophy_observer.record_result(found_game_result)

        button_pressed  = False
        end_screen_time = time.time()

        while current_state.startswith("end") and time.time() - end_screen_time < END_SCREEN_TIMEOUT:
            if self._scrcpy_disconnected():
                return

            if not button_pressed:
                if self.play_again_on_win and found_game_result == "victory":
                    self.window_controller.press_key("F")
                else:
                    self.window_controller.press_key("Q")
                    time.sleep(2)
                    self.window_controller.press_key("Q")
                button_pressed = True

            time.sleep(0.5)
            heartbeat.bump()
            screenshot = self._safe_screenshot()
            if screenshot is None:
                return
            current_state = get_state(screenshot)

        if current_state.startswith("end"):
            log.warning("Stuck on end screen, restarting Brawl Stars")
            self.window_controller.restart_brawl_stars()
            return

        if self.play_again_on_win and found_game_result == "victory":
            start_wait = time.time()
            while time.time() - start_wait < PLAY_AGAIN_TIMEOUT:
                if self._scrcpy_disconnected():
                    return
                screenshot = self._safe_screenshot()
                if screenshot is None:
                    return
                if get_state(screenshot) == "match":
                    return
                time.sleep(0.5)
                heartbeat.bump()

            log.warning("Match did not start within timeout, pressing Q")
            self.window_controller.press_key("Q")
            time.sleep(2)
            self.window_controller.press_key("Q")

    def quit_shop(self):
        self.window_controller.click(100, 60)

    def close_pop_up(self):
        screenshot = self._safe_screenshot()
        if screenshot is None:
            return
        if self.close_popup_icon is None:
            self.close_popup_icon = load_image(img_state("close_popup.png"))
        popup_location = find_template_center(screenshot, self.close_popup_icon)
        if popup_location:
            self.window_controller.click(*popup_location)

    def do_state(self, state: str):
        handler = self.states.get(state)
        if handler:
            handler()
        else:
            log.warning(f"Unknown state '{state}', ignoring")
