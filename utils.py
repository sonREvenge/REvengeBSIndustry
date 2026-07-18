import copy
import glob
import hashlib
import json
import os
import threading

import cv2
import requests
import toml

from brawl_industry_logger import get_logger

log = get_logger("brawl_industry.utils")

CONFIG_DIR = "cfg"

_toml_meta_lock = threading.Lock()
_toml_locks: dict[str, threading.Lock] = {}


def _get_lock(path: str) -> threading.Lock:
    with _toml_meta_lock:
        lock = _toml_locks.get(path)
        if lock is None:
            lock = threading.Lock()
            _toml_locks[path] = lock
        return lock


def resolve_cfg_path(file_path: str) -> str:
    normalized = file_path.replace("\\", "/")
    if normalized.startswith("./cfg/"):
        return os.path.join(CONFIG_DIR, normalized[6:])
    if normalized.startswith("cfg/"):
        return os.path.join(CONFIG_DIR, normalized[4:])
    return file_path


cached_toml: dict = {}


def load_toml_as_dict(file_path: str) -> dict:
    file_path = resolve_cfg_path(file_path)
    with _get_lock(file_path):
        if file_path not in cached_toml:
            if os.path.exists(file_path):
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        cached_toml[file_path] = toml.load(f)
                except Exception as e:
                    log.warning(f"Corrupted TOML {file_path}: {e}")
                    cached_toml[file_path] = {}
            else:
                cached_toml[file_path] = {}
        return copy.deepcopy(cached_toml[file_path])


def save_dict_as_toml(data: dict, file_path: str):
    file_path = resolve_cfg_path(file_path)
    with _get_lock(file_path):
        tmp_path = file_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            toml.dump(data, f)
        os.replace(tmp_path, file_path)
        cached_toml[file_path] = copy.deepcopy(data)


_ALLOWED_API_HOSTS = {"angelfirela.pythonanywhere.com"}


def _validate_api_url(url: str) -> str:
    if url == "localhost" or url in _ALLOWED_API_HOSTS:
        return url
    log.warning(f"Untrusted api_base_url '{url}', falling back to default")
    return "angelfirela.pythonanywhere.com"


cfg_api_base_url = load_toml_as_dict("cfg/general_config.toml").get("api_base_url", "default")
api_base_url = _validate_api_url(
    cfg_api_base_url if cfg_api_base_url != "default" else "angelfirela.pythonanywhere.com"
)
brawlers_info_file_path = os.path.join(CONFIG_DIR, "brawlers_info.json")


def count_hsv_pixels(cv_image, low_hsv, high_hsv) -> int:
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, low_hsv, high_hsv)
    return cv2.countNonZero(mask)


def find_template_center(main_img, template, threshold: float = 0.8):
    main_gray = cv2.cvtColor(main_img, cv2.COLOR_RGB2GRAY)
    tmpl_gray = (
        cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        if len(template.shape) == 3 and template.shape[2] == 3
        else template
    )

    w, h = tmpl_gray.shape[::-1]
    result = cv2.matchTemplate(main_gray, tmpl_gray, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val >= threshold:
        return max_loc[0] + w // 2, max_loc[1] + h // 2
    return False


def load_brawlers_info() -> dict:
    path = brawlers_info_file_path
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            log.warning(f"Failed to load brawlers info: {e}")
    return {}


def get_online_wall_model_hash():
    try:
        r = requests.get(f"https://{api_base_url}/get_wall_model_hash", timeout=10)
        if r.status_code == 200:
            return r.json().get("hash", "")
    except requests.RequestException as e:
        log.warning(f"Failed to fetch wall model hash: {e}")
    return None


def calculate_sha256(file_path: str) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            h.update(chunk)
    return h.hexdigest()


def current_wall_model_is_latest() -> bool:
    from _paths import MODEL_TILES
    if not os.path.exists(MODEL_TILES):
        return False
    return calculate_sha256(MODEL_TILES) == get_online_wall_model_hash()


def get_latest_wall_model_file():
    from _paths import MODEL_TILES
    try:
        r = requests.get(f"https://{api_base_url}/get_wall_model_file", timeout=30)
        if r.status_code == 200:
            with open(MODEL_TILES, "wb") as f:
                f.write(r.content)
            log.info("Downloaded the latest wall model")
        else:
            log.warning(f"Failed to download wall model, status: {r.status_code}")
    except requests.RequestException as e:
        log.warning(f"Failed to download wall model: {e}")


def get_latest_wall_model_classes():
    try:
        r = requests.get(f"https://{api_base_url}/get_wall_model_classes", timeout=10)
        if r.status_code == 200:
            return r.json().get("classes", [])
    except requests.RequestException as e:
        log.warning(f"Failed to fetch wall model classes: {e}")
    return None


def update_wall_model_classes():
    classes = get_latest_wall_model_classes()
    if not classes:
        return
    full_config     = load_toml_as_dict("cfg/bot_config.toml")
    current_classes = full_config.get("wall_model_classes", [])
    if classes == current_classes:
        return
    full_config["wall_model_classes"] = classes
    save_dict_as_toml(full_config, "cfg/bot_config.toml")
    log.info("Wall model classes updated")


_config_mtimes: dict = {}


def _get_toml_mtimes() -> dict:
    cfg_dir = resolve_cfg_path("cfg/")
    mtimes = {}
    for f in glob.glob(os.path.join(cfg_dir, "*.toml")):
        try:
            mtimes[f] = os.path.getmtime(f)
        except OSError:
            pass
    return mtimes


def start_config_watcher():
    import time as _t

    def _watcher():
        global _config_mtimes
        _config_mtimes = _get_toml_mtimes()
        while True:
            _t.sleep(5)
            new_mtimes = _get_toml_mtimes()
            for path, mtime in new_mtimes.items():
                if _config_mtimes.get(path) != mtime:
                    with _get_lock(path):
                        cached_toml.pop(path, None)
                    log.info(f"Config reloaded: {os.path.basename(path)}")
            _config_mtimes = new_mtimes

    threading.Thread(target=_watcher, daemon=True).start()
