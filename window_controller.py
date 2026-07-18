import atexit
import math
import threading
import time
from typing import List

import scrcpy
from adbutils import adb

from utils import load_toml_as_dict, save_dict_as_toml
from brawl_industry_logger import get_logger

log = get_logger("brawl_industry.emulator")

KNOWN_BS_PACKAGES = ("com.supercell.brawlstars", "bsd.suitcase.release")


def restart_adb_server() -> None:
    try:
        adb.server_kill()
    except Exception:
        pass
    time.sleep(0.5)
    try:
        adb.server_start()
    except Exception:
        pass
    time.sleep(0.5)


def _get_bluestacks_paths() -> tuple[str, str]:
    cfg = load_toml_as_dict("cfg/general_config.toml")
    conf = cfg.get("bluestacks_conf", r"C:\ProgramData\BlueStacks_nxt\bluestacks.conf")
    exe  = cfg.get("bluestacks_exe",  r"C:\Program Files\BlueStacks_nxt\HD-Player.exe")
    return conf, exe


BLUESTACKS_CONF, BLUESTACKS_EXE = _get_bluestacks_paths()


def _parse_bluestacks_conf() -> dict:
    result = {}
    try:
        with open(BLUESTACKS_CONF, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "=" in line:
                    key, _, val = line.partition("=")
                    result[key.strip()] = val.strip().strip('"')
    except Exception as e:
        log.warning(f"Could not read bluestacks.conf: {e}")
    return result


def _resolve_bluestacks_instance(config_port: int) -> tuple[str | None, int]:
    conf = _parse_bluestacks_conf()
    if not conf:
        return None, config_port

    instance_name = None
    for key, val in conf.items():
        if (key.startswith("bst.instance.")
                and key.endswith(".adb_port")
                and ".status." not in key):
            try:
                if int(val) == config_port:
                    instance_name = key.replace("bst.instance.", "").replace(".adb_port", "")
                    break
            except (ValueError, TypeError):
                continue

    if not instance_name:
        return None, config_port

    live_port_str = conf.get(f"bst.instance.{instance_name}.status.adb_port", "")
    try:
        live_port = int(live_port_str)
    except (ValueError, TypeError):
        live_port = config_port

    if live_port != config_port:
        log.warning(f"BlueStacks port changed: config={config_port} to live={live_port} (instance {instance_name})")

    return instance_name, live_port


key_coords_dict = {
    "H": (1400, 990),
    "G": (1640, 990),
    "M": (1725, 800),
    "Q": (1660, 980),
    "E": (1510, 880),
    "F": (1360, 920),
}

directions_xy_deltas_dict = {
    "w": (0,    -150),
    "a": (-150,    0),
    "s": (0,     150),
    "d": (150,     0),
}


class WindowController:

    def __init__(self):
        self.scale_factor = 1.0
        self.width        = 1920
        self.height       = 1080
        self.width_ratio  = 1.0
        self.height_ratio = 1.0
        self.joystick_x, self.joystick_y = 220, 870

        config = load_toml_as_dict("cfg/general_config.toml")
        config_port       = config.get("emulator_port", 5037)
        self.package_name = config.get("brawl_stars_package", "com.supercell.brawlstars")

        try:
            config_port = int(config_port)
            if not (1024 <= config_port <= 65535):
                log.warning(f"Port {config_port} out of range, using 5037")
                config_port = 5037
        except (ValueError, TypeError):
            log.warning("Invalid port value, using 5037")
            config_port = 5037

        self._bs_instance, self.port = _resolve_bluestacks_instance(config_port)
        if self._bs_instance:
            log.info(f"BlueStacks instance: {self._bs_instance} (port {self.port})")
        else:
            log.warning(f"BlueStacks instance not found for port {config_port}, connecting anyway")

        log.info(f"Connecting to ADB on port {self.port}")
        try:
            target_serial = f"127.0.0.1:{self.port}"
            self.device = self._find_device(target_serial)
            if not self.device:
                try:
                    adb.connect(target_serial)
                except Exception as e:
                    log.warning(f"ADB connect failed: {e}")
                self.device = self._find_device(target_serial)

            if not self.device:
                raise ConnectionError(f"No ADB device found for port {self.port}")

            log.info(f"Connected to device: {self.device.serial}")

            self.frame_lock       = threading.Lock()
            self.last_frame       = None
            self.last_frame_time  = 0.0
            self.frame_id         = 0
            self.last_joystick_pos = None
            self.FRAME_STALE_TIMEOUT = 10.0

            def on_frame(frame):
                if frame is not None:
                    with self.frame_lock:
                        self.last_frame      = frame
                        self.last_frame_time = time.time()
                        self.frame_id       += 1

            self._scrcpy_disconnected = threading.Event()

            def on_disconnect():
                log.warning("scrcpy disconnected")
                self._scrcpy_disconnected.set()

            self.scrcpy_client = scrcpy.Client(
                device=self.device,
                max_width=0,
                bitrate=2_000_000,
                max_fps=8,
            )
            self.scrcpy_client.add_listener(scrcpy.EVENT_FRAME, on_frame)
            self.scrcpy_client.add_listener(scrcpy.EVENT_DISCONNECT, on_disconnect)
            self.scrcpy_client.start(threaded=True)
            log.info("scrcpy client started")

            self._atexit_registered = True
            atexit.register(self.close)

        except Exception as e:
            raise ConnectionError(f"Failed to initialize scrcpy: {e}")

        self.are_we_moving     = False
        self.PID_JOYSTICK      = 1
        self.PID_ATTACK        = 2
        self._watchdog_running = True

        self.crash_check_interval = load_toml_as_dict("cfg/time_thresholds.toml").get(
            "check_if_brawl_stars_crashed", 30.0
        )
        self._crash_event = threading.Event()
        self._watchdog_thread = threading.Thread(target=self._watchdog_loop, daemon=True)
        self._watchdog_thread.start()

    @staticmethod
    def _find_device(target_serial: str):
        for d in adb.device_list():
            if d.serial == target_serial:
                return d
        return None

    @property
    def is_brawl_stars_crashed(self):
        return self._crash_event.is_set()

    @is_brawl_stars_crashed.setter
    def is_brawl_stars_crashed(self, value):
        if value:
            self._crash_event.set()
        else:
            self._crash_event.clear()

    def _watchdog_loop(self):
        while self._watchdog_running:
            time.sleep(self.crash_check_interval)
            try:
                app = self.device.app_current()
            except Exception:
                continue
            if not app:
                continue
            opened = app.package.strip()
            if opened == self.package_name:
                continue
            if opened in KNOWN_BS_PACKAGES:
                cfg = load_toml_as_dict("cfg/general_config.toml")
                cfg["brawl_stars_package"] = opened
                save_dict_as_toml(cfg, "cfg/general_config.toml")
                self.package_name = opened
                log.info(f"Detected Brawl Stars package '{opened}', updated config")
                continue
            self._crash_event.set()

    def get_latest_frame(self):
        with self.frame_lock:
            if self.last_frame is None:
                return None, 0.0, 0
            return self.last_frame, self.last_frame_time, self.frame_id

    def restart_brawl_stars(self):
        try:
            self.device.app_stop(self.package_name)
            time.sleep(1)
            self.device.app_start(self.package_name)
            time.sleep(3)
        except Exception as e:
            log.warning(f"Failed to restart Brawl Stars: {e}")
        self._crash_event.clear()
        with self.frame_lock:
            self.last_frame_time = time.time()
        log.info("Brawl Stars restarted")

    def restart_bluestacks(self):
        if not self._bs_instance:
            log.error("Cannot restart BlueStacks, instance name unknown")
            return False
        import subprocess
        log.info(f"Restarting BlueStacks instance {self._bs_instance}")
        try:
            subprocess.Popen([BLUESTACKS_EXE, "--instance", self._bs_instance])
            time.sleep(20)
            return True
        except Exception as e:
            log.error(f"Failed to launch BlueStacks: {e}")
            return False

    def screenshot(self):
        if self._crash_event.is_set():
            log.warning("Brawl Stars crashed, restarting")
            self.restart_brawl_stars()
            return None

        if self._scrcpy_disconnected.is_set():
            raise ConnectionError("scrcpy disconnected")
        if not self.scrcpy_client.alive:
            raise ConnectionError("scrcpy client is no longer alive")

        raw_frame, frame_time, _ = self.get_latest_frame()

        deadline = time.time() + 15
        while raw_frame is None:
            if time.time() > deadline:
                raise ConnectionError("No frame received from scrcpy within 15s")
            if self._scrcpy_disconnected.is_set() or not self.scrcpy_client.alive:
                raise ConnectionError("scrcpy died while waiting for first frame")
            time.sleep(0.1)
            raw_frame, frame_time, _ = self.get_latest_frame()

        if frame_time > 0 and time.time() - frame_time > self.FRAME_STALE_TIMEOUT:
            raise ConnectionError(f"scrcpy frame is {time.time() - frame_time:.1f}s stale")

        return raw_frame

    def _touch(self, action, x, y, pointer_id):
        if not self.scrcpy_client.alive:
            return
        try:
            self.scrcpy_client.control.touch(int(x), int(y), action, pointer_id)
        except Exception as e:
            log.warning(f"touch failed: {e}")

    def touch_down(self, x, y, pointer_id=0):
        self._touch(scrcpy.ACTION_DOWN, x, y, pointer_id)

    def touch_move(self, x, y, pointer_id=0):
        self._touch(scrcpy.ACTION_MOVE, x, y, pointer_id)

    def touch_up(self, x, y, pointer_id=0):
        self._touch(scrcpy.ACTION_UP, x, y, pointer_id)

    def keys_up(self, _keys: List[str] = None):
        if self.are_we_moving:
            self.touch_up(self.joystick_x, self.joystick_y, pointer_id=self.PID_JOYSTICK)
            self.are_we_moving     = False
            self.last_joystick_pos = None

    def keys_down(self, keys: List[str]):
        delta_x = delta_y = 0
        for key in keys:
            if key in directions_xy_deltas_dict:
                dx, dy = directions_xy_deltas_dict[key]
                delta_x += dx
                delta_y += dy

        new_pos = (
            self.joystick_x + delta_x * self.width_ratio,
            self.joystick_y + delta_y * self.height_ratio,
        )

        if not self.are_we_moving:
            self.touch_down(self.joystick_x, self.joystick_y, pointer_id=self.PID_JOYSTICK)
            self.are_we_moving = True
            self.last_joystick_pos = None

        if self.last_joystick_pos != new_pos:
            self.touch_move(new_pos[0], new_pos[1], pointer_id=self.PID_JOYSTICK)
            self.last_joystick_pos = new_pos

    def click(self, x, y, delay=0.005, already_include_ratio=True, touch_up=True, touch_down=True):
        if not already_include_ratio:
            x *= self.width_ratio
            y *= self.height_ratio
        if touch_down:
            self.touch_down(x, y, pointer_id=self.PID_ATTACK)
        if delay > 0:
            time.sleep(delay)
        if touch_up:
            self.touch_up(x, y, pointer_id=self.PID_ATTACK)

    def press_key(self, key, delay=0.005, touch_up=True, touch_down=True):
        if key not in key_coords_dict:
            return
        x, y = key_coords_dict[key]
        self.click(
            x * self.width_ratio, y * self.height_ratio,
            delay, touch_up=touch_up, touch_down=touch_down,
        )

    def swipe(self, start_x, start_y, end_x, end_y, duration=0.2):
        dist_x   = end_x - start_x
        dist_y   = end_y - start_y
        distance = math.sqrt(dist_x ** 2 + dist_y ** 2)
        if distance == 0:
            return
        steps      = max(int(distance / 25), 1)
        step_delay = duration / steps
        self.touch_down(int(start_x), int(start_y), pointer_id=self.PID_ATTACK)
        for i in range(1, steps + 1):
            time.sleep(step_delay)
            t = i / steps
            self.touch_move(
                int(start_x + dist_x * t),
                int(start_y + dist_y * t),
                pointer_id=self.PID_ATTACK,
            )
        self.touch_up(int(end_x), int(end_y), pointer_id=self.PID_ATTACK)

    def close(self):
        self._watchdog_running = False
        if hasattr(self, "scrcpy_client"):
            stopped = threading.Event()

            def _stop():
                try:
                    self.scrcpy_client.stop()
                except Exception as e:
                    log.warning(f"Error closing scrcpy client: {e}")
                finally:
                    stopped.set()

            t = threading.Thread(target=_stop, daemon=True)
            t.start()
            if not stopped.wait(timeout=2.0):
                log.warning("scrcpy_client.stop() timed out")

        if getattr(self, "_atexit_registered", False):
            try:
                atexit.unregister(self.close)
            except Exception:
                pass
            self._atexit_registered = False
