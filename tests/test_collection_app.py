from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import cv2

from human_tracker.annotation import BoxEditor, review_sample
from human_tracker.collection_app import capture_loop
from human_tracker.dataset import CaptureRun, read_image, read_manifest

from test_dataset import frame


class MockStream:
    def __init__(self, steps):
        self.steps = iter(steps)
        self.now = None
        self.state = None
        self.key = None
        self.tick()

    def tick(self):
        self.now, image, self.key = next(self.steps)
        self.state = SimpleNamespace(frame=image, error=None, status='Streaming' if image else 'Reconnecting')


class CaptureLoopTests(unittest.TestCase):
    def run_loop(self, root, steps, *, interval=None, count=None):
        stream = MockStream(steps)
        run = CaptureRun(root, 'test')
        shown = []

        def wait_key(_):
            key = stream.key
            if key != ord('q'):
                stream.tick()
            return key

        with patch.multiple(cv2, imshow=lambda _name, image: shown.append(image.copy()),
                            waitKey=wait_key, getWindowProperty=lambda *_: 1):
            saved = capture_loop(cv2, stream, run, interval=interval, count=count, clock=lambda: stream.now)
        return saved, shown

    def test_ten_manual_frames_are_raw_and_disconnect_drops_displayed_frame(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = [(10, frame(200, received_at=10), 32), (10.1, None, -1)]
            steps += [(11 + n / 100, frame(n, connection=2, received_at=11 + n / 100), 32)
                      for n in range(1, 11)]
            steps += [(11.2, frame(11, connection=2, received_at=11.2), ord('q'))]
            saved, shown = self.run_loop(root, steps, count=10)
            self.assertEqual(saved, 10)
            self.assertTrue(shown[0].any())
            saved_frames = [read_image(row.image_path(root)) for row in read_manifest(root)]
            self.assertCountEqual([image.tobytes() for image in saved_frames],
                                  [frame(n).image.tobytes() for n in range(1, 11)])

    def test_ten_timed_frames_respect_interval_and_resume_on_a_new_connection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            steps = [(10, frame(200, received_at=10), -1), (10.1, None, -1)]
            steps += [(11 + n, frame(n + 1, connection=2, received_at=11 + n), -1) for n in range(22)]
            steps += [(34, None, ord('q'))]
            saved, _ = self.run_loop(root, steps, interval=2, count=10)
            self.assertEqual(saved, 10)
            saved_frames = [read_image(row.image_path(root)) for row in read_manifest(root)]
            self.assertCountEqual([image.tobytes() for image in saved_frames],
                                  [frame(n).image.tobytes() for n in range(3, 22, 2)])


class AnnotationTests(unittest.TestCase):
    def test_drag_uses_native_coordinates_and_supports_all_frame_edges(self):
        editor = BoxEditor()
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 639, 479, 0, None)
        editor.mouse(cv2.EVENT_MOUSEMOVE, 0, 0, 0, None)
        editor.mouse(cv2.EVENT_LBUTTONUP, 0, 0, 0, None)
        self.assertEqual(editor.boxes, [(0, 0, 320, 240)])
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 100, 100, 0, None)
        editor.mouse(cv2.EVENT_LBUTTONUP, 100, 100, 0, None)
        self.assertEqual(len(editor.boxes), 1)
        editor.mouse(cv2.EVENT_LBUTTONDOWN, 0, 0, 0, None)
        editor.mouse(cv2.EVENT_LBUTTONUP, 639, 479, 0, None)
        self.assertEqual(len(editor.boxes), 1)

    def test_review_sampling_covers_each_split_and_both_label_states(self):
        rows = [(SimpleNamespace(split=split), positive)
                for split in ('train', 'val', 'test') for positive in (True, False) for _ in range(4)]
        sample = review_sample(rows, 6)
        self.assertCountEqual([(row.split, positive) for row, positive in sample],
                              [(split, positive) for split in ('train', 'val', 'test') for positive in (True, False)])


if __name__ == '__main__':
    unittest.main()
