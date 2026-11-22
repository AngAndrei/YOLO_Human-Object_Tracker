from types import SimpleNamespace
import unittest
from unittest.mock import patch

from phase2_accuracy import score_accuracy, wait_for_frame


class AccuracyTests(unittest.TestCase):
    def test_counts_reviewed_person_boxes_and_empty_false_positive_frames(self):
        occupied = [f"person-{i}" for i in range(10)]
        empty = [f"empty-{i}" for i in range(10)]
        calls = []

        def detect(image):
            calls.append(image)
            return ["box", "second box"] if image in occupied[:9] + empty[:1] else []

        reviewed = []

        def review(index, image, detections):
            reviewed.append(image)
            return index != 9

        self.assertEqual(score_accuracy(detect, occupied, empty, review), (8, 1))
        self.assertEqual(calls, occupied + empty)
        self.assertEqual(reviewed, occupied)

    def test_requires_both_complete_sample_sets(self):
        with self.assertRaisesRegex(ValueError, "10"):
            score_accuracy(None, [None] * 10, [None] * 9, None)

    def test_wait_for_frame_rejects_capture_from_a_new_connection(self):
        previous = SimpleNamespace(sequence=12, connection_id=3)
        current = SimpleNamespace(sequence=1, connection_id=4)
        worker = SimpleNamespace(state=SimpleNamespace(frame=current, error=None, status="Streaming"))
        with self.assertRaisesRegex(RuntimeError, "connection changed"):
            wait_for_frame(worker, previous=previous)

    def test_wait_for_frame_ignores_a_repeated_sequence(self):
        old = SimpleNamespace(sequence=12, connection_id=3)
        new = SimpleNamespace(sequence=13, connection_id=3)
        worker = SimpleNamespace(state=SimpleNamespace(frame=old, error=None, status="Streaming"))

        def next_state(_):
            worker.state.frame = new

        with patch("phase2_accuracy.time.sleep", side_effect=next_state):
            self.assertIs(wait_for_frame(worker, previous=old), new)
