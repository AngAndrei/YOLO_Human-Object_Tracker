import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from human_tracker.detector import PersonDetector
from human_tracker.detection_app import current_result, main


class Boxes(list):
    def tolist(self):
        return list(self)


class FakeModel:
    task = "detect"
    names = {0: "cat", 4: "person"}

    def predict(self, **kwargs):
        self.arguments = kwargs
        return [SimpleNamespace(boxes=SimpleNamespace(data=Boxes([
            [-5, -1, 345, 251, 0.9, 4],
            [20, 20, 40, 40, 0.99, 0],
            [20, 20, 40, 40, 0.2, 4],
            [40, 20, 40, 40, 0.99, 4],
        ])))]


class PersonDetectionTests(unittest.TestCase):
    def test_filters_and_clips_boxes_to_source_image(self):
        model = FakeModel()
        detector = PersonDetector(model)
        frame = SimpleNamespace(shape=(240, 320, 3))
        detections = detector.detect(frame)
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].box, (0, 0, 319, 239))
        self.assertEqual(detections[0].confidence, 0.9)
        self.assertEqual(model.arguments["classes"], [4])
        self.assertEqual(model.arguments["device"], "cpu")
        self.assertEqual(model.arguments["imgsz"], 320)

    def test_rejects_model_without_person_class(self):
        with self.assertRaisesRegex(ValueError, "person"):
            PersonDetector(SimpleNamespace(task="detect", names={0: "cat"}))

    def test_prediction_at_display_cutoff_survives_strict_model_filter(self):
        model = FakeModel()
        frame = SimpleNamespace(shape=(240, 320, 3))
        detections = PersonDetector(model, confidence=0.9).detect(frame)
        self.assertLess(model.arguments['conf'], 0.9)
        self.assertEqual(detections[0].confidence, 0.9)

    def test_only_publishes_when_result_matches_active_connection(self):
        result = SimpleNamespace(connection_id=3, sequence=10)
        self.assertTrue(current_result(result, SimpleNamespace(connection_id=3, sequence=12)))
        self.assertFalse(current_result(result, SimpleNamespace(connection_id=4, sequence=13)))
        self.assertFalse(current_result(result, None))

    def test_room_objects_selects_only_the_three_named_classes_in_one_call(self):
        from human_tracker.detector import ObjectDetector

        model = FakeModel()
        model.names = {4: "person", 6: "bottle", 11: "book", 18: "cup"}
        rows = [
            [5, 7, 14, 28, 0.9, 6],
            [-2, 10, 50, 35, 0.7, 11],
            [8, 8, 23, 23, 0.6, 18],
            [0, 0, 35, 40, 0.99, 4],
            [10, 40, 70, 55, 0.2, 6],
        ]
        prediction = [SimpleNamespace(boxes=SimpleNamespace(data=Boxes(rows)))]
        with patch.object(model, "predict", return_value=prediction) as predict:
            detector = ObjectDetector(model, classes=("bottle", "book", "cup"))
            detections = detector.detect(SimpleNamespace(shape=(240, 320, 3)))
        self.assertEqual([item.class_name for item in detections], ["bottle", "book", "cup"])
        self.assertEqual(detections[1].box, (0, 10, 50, 35))
        predict.assert_called_once()
        self.assertEqual(predict.call_args.kwargs["classes"], [6, 11, 18])

    def test_room_object_replay_reports_class_counts(self):
        import numpy as np

        from human_tracker.detector import ObjectDetector

        model = FakeModel()
        model.names = {0: "cat", 4: "person", 5: "cup", 6: "bottle", 8: "book"}
        detector = ObjectDetector(model, classes=("bottle", "book", "cup"))
        output = io.StringIO()
        with patch("human_tracker.detection_app.load_detector", return_value=detector) as load:
            with patch("human_tracker.detection_app.replay_frames", return_value=(
                frame for frame in [np.zeros((240, 320, 3), dtype=np.uint8)]
            )), redirect_stdout(output):
                status = main("replay", ["--source", "desk.png", "--room-objects", "--headless"])
        self.assertEqual(status, 0)
        self.assertTrue(load.call_args.kwargs["room_objects"])
        self.assertIn("0 bottle boxes, 0 book boxes, 0 cup boxes", output.getvalue())
