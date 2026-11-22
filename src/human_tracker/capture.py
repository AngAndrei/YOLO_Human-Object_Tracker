"""Read JPEG frames from the XIAO's native USB serial port."""

from dataclasses import dataclass
from datetime import datetime, timezone
import struct
import time

MARKER = b"XIAO"
HEADER = struct.Struct("<4sI")
MAX_JPEG_BYTES = 128 * 1024
JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"


class JpegFramer:
    def __init__(self):
        self.buffer = bytearray()

    def _end(self, start):
        if len(self.buffer) < start + HEADER.size + len(JPEG_START):
            return None
        _, size = HEADER.unpack_from(self.buffer, start)
        payload = start + HEADER.size
        if not 4 <= size <= MAX_JPEG_BYTES or not self.buffer.startswith(JPEG_START, payload):
            return -1
        return payload + size

    def _next_complete_frame(self):
        index = self.buffer.find(MARKER, HEADER.size)
        while index >= 0:
            end = self._end(index)
            if end is not None and 0 < end <= len(self.buffer) and self.buffer[end - 2:end] == JPEG_END:
                return index
            index = self.buffer.find(MARKER, index + len(MARKER))
        return -1

    def feed(self, chunk):
        self.buffer.extend(chunk)
        frames = []
        while self.buffer:
            start = self.buffer.find(MARKER)
            if start < 0:
                del self.buffer[:-(len(MARKER) - 1)]
                break
            del self.buffer[:start]
            end = self._end(0)
            if end == -1:
                del self.buffer[:len(MARKER)]
                continue
            if end is None:
                break
            if end > len(self.buffer):
                next_start = self._next_complete_frame()
                if next_start < 0:
                    break
                del self.buffer[:next_start]
                continue
            if self.buffer[end - 2:end] == JPEG_END:
                frames.append(bytes(self.buffer[HEADER.size:end]))
                del self.buffer[:end]
            else:
                del self.buffer[:len(MARKER)]
        return frames


@dataclass(frozen=True)
class Frame:
    image: object
    sequence: int
    connection_id: int
    received_at: float
    received_utc: datetime | None = None


class UsbCamera:
    def __init__(self, connect, decode, *, clock=time.monotonic):
        self._connect = connect
        self._decode = decode
        self._clock = clock
        self._port = None
        self._framer = JpegFramer()
        self._last_frame_at = 0.0
        self._retry_at = 0.0
        self._retry_delay = 0.25
        self.connection_id = 0
        self.decoded_count = 0
        self.latest = None
        self.status = "Waiting for USB camera"

    def _retry(self, reason):
        self.close()
        self.status = f"Reconnecting: {reason}"
        self._retry_at = self._clock() + self._retry_delay
        self._retry_delay = min(self._retry_delay * 2, 2.0)

    def close(self):
        port, self._port = self._port, None
        self.latest = None
        self._framer.buffer.clear()
        if port is not None:
            try:
                port.close()
            except OSError:
                pass

    def poll(self):
        if self._port is None:
            if self._clock() < self._retry_at:
                return
            try:
                self._port = self._connect()
            except OSError as exc:
                self._retry(exc)
                return
            self.connection_id += 1
            self._last_frame_at = self._clock()
            self.status = "Waiting for JPEG frames"

        try:
            chunk = self._port.read(8192)
            for jpeg in self._framer.feed(chunk):
                image = self._decode(jpeg)
                if image is not None:
                    self._last_frame_at = self._clock()
                    self._retry_delay = 0.25
                    self.decoded_count += 1
                    self.latest = Frame(image, self.decoded_count, self.connection_id,
                                        self._last_frame_at, datetime.now(timezone.utc))
                    self.status = "Streaming"
        except OSError as exc:
            self._retry(exc)
            return

        if self._clock() - self._last_frame_at >= 2:
            self._retry("No valid JPEG frames for 2 seconds")


def decode_jpeg(jpeg):
    import cv2
    import numpy as np

    try:
        image = cv2.imdecode(np.frombuffer(jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    except cv2.error:
        return None
    if image is not None and image.shape == (240, 320, 3):
        return image
    return None


def open_camera(port_name):
    import serial

    return UsbCamera(lambda: serial.Serial(port_name, baudrate=115200, timeout=0.04), decode_jpeg)
