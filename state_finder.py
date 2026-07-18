import os
import cv2
from utils import load_toml_as_dict
from brawl_industry_logger import get_logger

log = get_logger("brawl_industry.state")

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

_gen_cfg = load_toml_as_dict("cfg/general_config.toml")

states_path      = os.path.join(_BASE_DIR, "images", "states") + os.sep
star_drops_path  = os.path.join(_BASE_DIR, "images", "star_drop_types") + os.sep
end_results_path = os.path.join(_BASE_DIR, "images", "end_results") + os.sep

images_with_star_drop = [f for f in os.listdir(star_drops_path) if "star_drop" in f]

_lobby_cfg  = load_toml_as_dict("./cfg/lobby_config.toml")
region_data = _lobby_cfg.get("template_matching", {})
crop_region = _lobby_cfg.get("lobby", {}).get("trophy_observer", [0, 0, 1920, 200])

_DEBUG = _gen_cfg.get("super_debug", "no") == "yes"

_debug_folder     = os.path.join(_BASE_DIR, "debug_frames") + os.sep
_MAX_DEBUG_FRAMES = 100

if _DEBUG and not os.path.exists(_debug_folder):
    os.makedirs(_debug_folder)


cached_templates = {}


def load_template(image_path):
    if image_path in cached_templates:
        return cached_templates[image_path]
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Template not found: '{image_path}'. Run the bot from the project root."
        )
    cached_templates[image_path] = image
    return image


def is_template_in_region(image, template_path, region, threshold: float = 0.7):
    orig_x, orig_y, orig_width, orig_height = region
    cropped  = image[orig_y:orig_y + orig_height, orig_x:orig_x + orig_width]
    template = load_template(template_path)
    result   = cv2.matchTemplate(cropped, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val > threshold


_SHOWDOWN_PLACE_THRESHOLD = 0.9
_SHOWDOWN_TEMPLATES = (
    ("1st.png",     "victory"),
    ("2nd.png",     "draw"),
    ("3rd.png",     "defeat"),
    ("3rd_alt.png", "defeat"),
    ("4th.png",     "defeat"),
)


def find_game_result(screenshot):
    for template_name, mapped_result in _SHOWDOWN_TEMPLATES:
        if is_template_in_region(
            screenshot, end_results_path + template_name, crop_region,
            threshold=_SHOWDOWN_PLACE_THRESHOLD,
        ):
            return mapped_result
    if is_template_in_region(screenshot, end_results_path + "victory.png", crop_region): return "victory"
    if is_template_in_region(screenshot, end_results_path + "defeat.png",  crop_region): return "defeat"
    if is_template_in_region(screenshot, end_results_path + "draw.png",    crop_region): return "draw"
    return None


_REGION_DEFAULTS = {
    "powerpoint":        [1000, 5,    80,  80],
    "brawler_menu_task": [1450, 0,    80,  80],
    "close_popup":       [1740, 140, 140, 100],
    "lobby_menu":        [1790, 20,   75,  65],
    "brawl_pass_house":  [1750, 0,   169, 100],
    "go_back_arrow":     [0,    0,   175, 110],
    "star_drop":         [790,  350, 350, 350],
    "trophies_screen":   [1545, 915, 365, 168],
    "exit_match_making": [1600, 925, 295, 135],
    "prestige_continue": [535,  950, 345,  95],
    "nano_noodles":      [360,  880, 215, 150],
}


def _region(key):
    return region_data.get(key, _REGION_DEFAULTS.get(key, [0, 0, 1920, 1080]))


def _match(image, template, key):
    return is_template_in_region(image, states_path + template, _region(key))


def is_in_shop(image):               return _match(image, "powerpoint.png",        "powerpoint")
def is_in_brawler_selection(image):  return _match(image, "brawler_menu_task.png", "brawler_menu_task")
def is_in_offer_popup(image):        return _match(image, "close_popup.png",       "close_popup")
def is_in_lobby(image):              return _match(image, "lobby_menu.png",        "lobby_menu")
def is_in_end_of_a_match(image):     return find_game_result(image)
def is_in_trophy_reward(image):      return _match(image, "trophies_screen.png",   "trophies_screen")
def is_in_brawl_pass(image):         return _match(image, "brawl_pass_house.png",  "brawl_pass_house")
def is_in_star_road(image):          return _match(image, "go_back_arrow.png",     "go_back_arrow")
def is_in_match_making(image):       return _match(image, "exit_match_making.png", "exit_match_making")
def is_in_prestige_milestone(image): return _match(image, "prestige_continue.png", "prestige_continue")
def is_in_nano_noodles(image):       return _match(image, "nano_noodles.png",      "nano_noodles")


def is_in_star_drop(image):
    for f in images_with_star_drop:
        if is_template_in_region(image, star_drops_path + f, _region("star_drop")):
            return True
    return False


def get_in_game_state(image):
    game_result = is_in_end_of_a_match(image)
    if game_result: return f"end_{game_result}"
    if is_in_shop(image):               return "shop"
    if is_in_offer_popup(image):        return "popup"
    if is_in_lobby(image):              return "lobby"
    if is_in_match_making(image):       return "match_making"
    if is_in_brawler_selection(image):  return "brawler_selection"
    if is_in_brawl_pass(image) or is_in_star_road(image): return "shop"
    if is_in_prestige_milestone(image): return "prestige_milestone"
    if is_in_nano_noodles(image):       return "nano_noodles"
    if is_in_star_drop(image):          return "star_drop"
    if is_in_trophy_reward(image):      return "trophy_reward"
    return "match"


def get_state(screenshot):
    screenshot_bgr = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    if _DEBUG:
        existing = sorted(os.listdir(_debug_folder))
        while len(existing) >= _MAX_DEBUG_FRAMES:
            try:
                os.remove(os.path.join(_debug_folder, existing.pop(0)))
            except OSError:
                break
        cv2.imwrite(
            os.path.join(_debug_folder, f"state_screenshot_{len(existing)}.png"),
            screenshot_bgr,
        )
    return get_in_game_state(screenshot_bgr)
