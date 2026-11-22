from collections import deque
from importlib.util import find_spec
import struct
import unittest

from human_tracker.capture import JpegFramer, UsbCamera

JPEG = b"\xff\xd8test image bytes\xff\xd9"


def packet(payload=JPEG):
    return b"XIAO" + struct.pack("<I", len(payload)) + payload


class FramingTests(unittest.TestCase):
    def test_fragmented_header_and_payload(self):
        framer = JpegFramer()
        wire = packet()
        self.assertEqual(framer.feed(b"startup" + wire[:2]), [])
        self.assertEqual(framer.feed(wire[2:11]), [])
        self.assertEqual(framer.feed(wire[11:]), [JPEG])

    def test_resynchronizes_after_invalid_lengths_and_jpeg(self):
        framer = JpegFramer()
        invalid = b"XIAO\x01\x00\x02\x00" + packet(b"not a jpeg")
        self.assertEqual(framer.feed(invalid + packet()), [JPEG])

    def test_resynchronizes_after_truncated_payload(self):
        framer = JpegFramer()
        truncated = b"XIAO" + struct.pack("<I", 60_000) + JPEG[:5]
        self.assertEqual(framer.feed(truncated), [])
        self.assertEqual(framer.feed(packet()), [JPEG])

    def test_keeps_marker_like_bytes_inside_complete_frame(self):
        framer = JpegFramer()
        jpeg = b"\xff\xd8 XIAO\x00\x00\x02\x00 comment \xff\xd9"
        self.assertEqual(framer.feed(packet(jpeg) + packet()), [jpeg, JPEG])


class Port:
    def __init__(self, *reads):
        self.reads = deque(reads)
        self.closed = False

    def read(self, size):
        result = self.reads.popleft() if self.reads else b""
        if isinstance(result, Exception):
            raise result
        return result

    def close(self):
        self.closed = True


class CameraTests(unittest.TestCase):
    def setUp(self):
        self.time = 0.0

    def camera(self, *ports):
        pending = deque(ports)

        def connect():
            port = pending.popleft()
            if isinstance(port, Exception):
                raise port
            return port

        return UsbCamera(connect, lambda jpeg: ("decoded", jpeg), clock=lambda: self.time)

    def test_disconnect_clears_latest_and_reconnects_with_new_id(self):
        before = Port(packet(), OSError("USB disconnected"))
        camera = self.camera(before, Port(packet()))
        camera.poll()
        first = camera.latest
        self.assertEqual(first.image, ("decoded", JPEG))
        camera.poll()
        self.assertIsNone(camera.latest)
        self.assertTrue(before.closed)
        self.time += 1
        camera.poll()
        self.assertNotEqual(camera.latest.connection_id, first.connection_id)
        self.assertEqual(camera.decoded_count, 2)
        camera.close()
        self.assertIsNone(camera.latest)

    def test_two_second_stall_retires_a_frame(self):
        port = Port(packet())
        camera = self.camera(port)
        camera.poll()
        self.time = 2.0
        camera.poll()
        self.assertIsNone(camera.latest)
        self.assertTrue(port.closed)

    def test_busy_port_reports_reason_and_retries(self):
        camera = self.camera(OSError("Access is denied"), Port(packet()))
        camera.poll()
        self.assertIn("Access is denied", camera.status)
        self.assertIsNone(camera.latest)
        self.time += 1
        camera.poll()
        self.assertIsNotNone(camera.latest)

    @unittest.skipUnless(find_spec("cv2"), "OpenCV is required for the JPEG integration test")
    def test_decodes_fragmented_320x240_jpeg_in_bgr(self):
        import cv2
        import numpy as np
        from human_tracker.capture import decode_jpeg

        red = np.full((240, 320, 3), (0, 0, 255), dtype=np.uint8)
        encoded, jpeg = cv2.imencode(".jpg", red)
        self.assertTrue(encoded)
        wire = packet(jpeg.tobytes())
        malformed = bytearray(jpeg.tobytes())
        sof = malformed.index(b"\xff\xc0")
        malformed[sof + 5:sof + 9] = struct.pack(">HH", 40_000, 40_000)
        port = Port(packet(malformed) + wire[:13], wire[13:])
        camera = UsbCamera(lambda: port, decode_jpeg)
        camera.poll()
        self.assertIsNone(camera.latest)
        camera.poll()
        frame = camera.latest.image
        self.assertEqual(frame.shape, (240, 320, 3))
        self.assertGreater(int(frame[120, 160, 2]), 250)
        self.assertLess(int(frame[120, 160, 0]), 5)


if __name__ == "__main__":
    unittest.main()
