import os
import socket
import struct
import threading
import time
from time import sleep
from typing import Any, Callable, Optional, Tuple, Union

import cv2
import numpy as np
from av.codec import CodecContext
from adbutils import AdbDevice, AdbError, adb

# adbutils 2.x moved Network and AdbConnection to _adb
try:
    from adbutils import Network, AdbConnection
except ImportError:
    from adbutils._adb import Network, AdbConnection

from .const import (
    EVENT_FRAME, EVENT_INIT, EVENT_DISCONNECT,
    LOCK_SCREEN_ORIENTATION_UNLOCKED,
)
from .control import ControlSender


class Client:
    def __init__(
        self,
        device: Optional[Union[AdbDevice, str]] = None,
        max_width: int = 0,
        bitrate: int = 8000000,
        max_fps: int = 0,
        flip: bool = False,
        block_frame: bool = False,
        stay_awake: bool = False,
        lock_screen_orientation: int = LOCK_SCREEN_ORIENTATION_UNLOCKED,
        connection_timeout: int = 3000,
    ):
        if device is None:
            device = adb.device_list()[0]
        elif isinstance(device, str):
            device = adb.device(serial=device)

        self.device = device
        self.listeners: dict = {EVENT_FRAME: [], EVENT_INIT: [], EVENT_DISCONNECT: []}

        self.last_frame: Optional[np.ndarray] = None
        self.resolution: Optional[Tuple[int, int]] = None
        self.device_name: Optional[str] = None
        self.control = ControlSender(self)

        self.flip = flip
        self.max_width = max_width
        self.bitrate = bitrate
        self.max_fps = max_fps
        self.block_frame = block_frame
        self.stay_awake = stay_awake
        self.lock_screen_orientation = lock_screen_orientation
        self.connection_timeout = connection_timeout

        self.alive = False
        self.__server_stream: Optional[AdbConnection] = None
        self.__video_socket: Optional[socket.socket] = None
        self.control_socket: Optional[socket.socket] = None
        self.control_socket_lock = threading.Lock()

    def __init_server_connection(self) -> None:
        for _ in range(self.connection_timeout // 100):
            try:
                self.__video_socket = self.device.create_connection(
                    Network.LOCAL_ABSTRACT, "scrcpy"
                )
                break
            except AdbError:
                sleep(0.1)
        else:
            raise ConnectionError("Failed to connect scrcpy-server after timeout")

        dummy_byte = self.__video_socket.recv(1)
        if not len(dummy_byte) or dummy_byte != b"\x00":
            raise ConnectionError("Did not receive Dummy Byte")

        self.control_socket = self.device.create_connection(
            Network.LOCAL_ABSTRACT, "scrcpy"
        )
        self.device_name = self.__video_socket.recv(64).decode("utf-8").rstrip("\x00")
        if not len(self.device_name):
            raise ConnectionError("Did not receive Device Name")

        res = self.__video_socket.recv(4)
        self.resolution = struct.unpack(">HH", res)
        self.__video_socket.setblocking(False)

        for sock in (self.__video_socket, self.control_socket):
            try:
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass

    def __deploy_server(self) -> None:
        jar_name = "scrcpy-server-v1.24.jar"
        server_file_path = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), jar_name
        )
        self.device.push(server_file_path, f"/data/local/tmp/{jar_name}")
        commands = [
            f"CLASSPATH=/data/local/tmp/{jar_name}",
            "app_process",
            "/",
            "com.genymobile.scrcpy.Server",
            "1.24",
            "log_level=info",
            f"bit_rate={self.bitrate}",
            f"max_size={self.max_width}",
            f"max_fps={self.max_fps}",
            f"lock_video_orientation={self.lock_screen_orientation}",
            "tunnel_forward=true",
            "control=true",
            "display_id=0",
            "show_touches=false",
            f"stay_awake={str(self.stay_awake).lower()}",
            "clipboard_autosync=false",
        ]
        self.__server_stream = self.device.shell(commands, stream=True)
        try:
            self.__server_stream.read(10)
        except Exception:
            pass

    def start(self, threaded: bool = False) -> None:
        assert self.alive is False

        self.__deploy_server()
        self.__init_server_connection()
        self.alive = True
        self.__send_to_listeners(EVENT_INIT)

        if threaded:
            t = threading.Thread(target=self.__stream_loop, daemon=True)
            t.start()
        else:
            self.__stream_loop()

    def stop(self) -> None:
        self.alive = False
        for s in (self.__server_stream, self.control_socket, self.__video_socket):
            if s is not None:
                try:
                    s.close()
                except Exception:
                    pass

    def __stream_loop(self) -> None:
        from av.error import InvalidDataError
        codec = CodecContext.create("h264", "r")
        try:
            while self.alive:
                try:
                    raw_h264 = self.__video_socket.recv(0x100000)
                except BlockingIOError:
                    time.sleep(0.01)
                    if not self.block_frame:
                        self.__send_to_listeners(EVENT_FRAME, None)
                    continue

                if not raw_h264:
                    break

                try:
                    packets = codec.parse(raw_h264)
                except InvalidDataError:
                    continue

                latest = None
                for packet in packets:
                    try:
                        frames = codec.decode(packet)
                    except InvalidDataError:
                        continue
                    if frames:
                        latest = frames[-1]

                if latest is None:
                    continue

                frame = latest.to_ndarray(format="rgb24")
                if self.flip:
                    frame = cv2.flip(frame, 1)
                self.last_frame = frame
                self.resolution = (frame.shape[1], frame.shape[0])
                self.__send_to_listeners(EVENT_FRAME, frame)
        except OSError:
            pass
        finally:
            if self.alive:
                self.alive = False
                self.__send_to_listeners(EVENT_DISCONNECT)

    def add_listener(self, cls: str, listener: Callable[..., Any]) -> None:
        if cls not in self.listeners:
            self.listeners[cls] = []
        self.listeners[cls].append(listener)

    def remove_listener(self, cls: str, listener: Callable[..., Any]) -> None:
        self.listeners.get(cls, []).remove(listener)

    def __send_to_listeners(self, cls: str, *args, **kwargs) -> None:
        for fn in self.listeners.get(cls, []):
            try:
                fn(*args, **kwargs)
            except Exception:
                pass
