import functools
import socket
import struct
from time import sleep

from scrcpy import const


def inject(control_type: int):
    def wrapper(f):
        @functools.wraps(f)
        def inner(*args, **kwargs):
            package = struct.pack(">B", control_type) + f(*args, **kwargs)
            if args[0].parent.control_socket is not None:
                with args[0].parent.control_socket_lock:
                    args[0].parent.control_socket.send(package)
            return package
        return inner
    return wrapper


class ControlSender:
    def __init__(self, parent):
        self.parent = parent

    @inject(const.TYPE_INJECT_KEYCODE)
    def keycode(self, keycode: int, action: int = const.ACTION_DOWN, repeat: int = 0) -> bytes:
        return struct.pack(">Biii", action, keycode, repeat, 0)

    @inject(const.TYPE_INJECT_TEXT)
    def text(self, text: str) -> bytes:
        buffer = text.encode("utf-8")
        return struct.pack(">i", len(buffer)) + buffer

    @inject(const.TYPE_INJECT_TOUCH_EVENT)
    def touch(self, x: int, y: int, action: int = const.ACTION_DOWN, touch_id: int = -1) -> bytes:
        x, y = max(x, 0), max(y, 0)
        return struct.pack(
            ">BqiiHHHi",
            action,
            touch_id,
            int(x),
            int(y),
            int(self.parent.resolution[0]),
            int(self.parent.resolution[1]),
            0xFFFF,
            1,
        )

    @inject(const.TYPE_INJECT_SCROLL_EVENT)
    def scroll(self, x: int, y: int, h: int, v: int) -> bytes:
        x, y = max(x, 0), max(y, 0)
        return struct.pack(
            ">iiHHii",
            int(x), int(y),
            int(self.parent.resolution[0]),
            int(self.parent.resolution[1]),
            int(h), int(v),
        )

    @inject(const.TYPE_BACK_OR_SCREEN_ON)
    def back_or_turn_screen_on(self, action: int = const.ACTION_DOWN) -> bytes:
        return struct.pack(">B", action)

    @inject(const.TYPE_EXPAND_NOTIFICATION_PANEL)
    def expand_notification_panel(self) -> bytes:
        return b""

    @inject(const.TYPE_EXPAND_SETTINGS_PANEL)
    def expand_settings_panel(self) -> bytes:
        return b""

    @inject(const.TYPE_COLLAPSE_PANELS)
    def collapse_panels(self) -> bytes:
        return b""

    @inject(const.TYPE_SET_SCREEN_POWER_MODE)
    def set_screen_power_mode(self, mode: int = const.POWER_MODE_NORMAL) -> bytes:
        return struct.pack(">b", mode)

    @inject(const.TYPE_ROTATE_DEVICE)
    def rotate_device(self) -> bytes:
        return b""

    def swipe(self, start_x, start_y, end_x, end_y, move_step_length=5, move_steps_delay=0.005):
        self.touch(start_x, start_y, const.ACTION_DOWN)
        next_x, next_y = start_x, start_y
        if end_x > self.parent.resolution[0]:
            end_x = self.parent.resolution[0]
        if end_y > self.parent.resolution[1]:
            end_y = self.parent.resolution[1]
        decrease_x = start_x > end_x
        decrease_y = start_y > end_y
        while True:
            if decrease_x:
                next_x = max(next_x - move_step_length, end_x)
            else:
                next_x = min(next_x + move_step_length, end_x)
            if decrease_y:
                next_y = max(next_y - move_step_length, end_y)
            else:
                next_y = min(next_y + move_step_length, end_y)
            self.touch(next_x, next_y, const.ACTION_MOVE)
            if next_x == end_x and next_y == end_y:
                self.touch(next_x, next_y, const.ACTION_UP)
                break
            sleep(move_steps_delay)
