from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from human_tracker import room_evaluation as room
from human_tracker.capture import Frame
from human_tracker.dataset import CaptureEntry, CaptureRun, DatasetError, read_label, write_label
from human_tracker.evaluation import FramePredictions, Pick


BOX = (10, 10, 50, 50)
OTHER = (60, 10, 100, 50)
CLASSES = ('bottle', 'book', 'cup')


class RoomEvaluationTests(unittest.TestCase):
    def test_multiclass_labels_preserve_class_and_person_format(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            row = CaptureEntry('frame', 'session', 'frame.png', '', 320, 240, ())
            write_label(root, row, [BOX])
            room_path = root / 'room-labels/frame.txt'
            write_label(root, row, [(1, BOX), (2, BOX)], classes=CLASSES)
            self.assertEqual([entry[0] for entry in read_label(room_path, classes=CLASSES)], [1, 2])
            self.assertEqual(len(read_label(row.label_path(root))), 1)
            self.assertTrue(row.label_path(root).read_text().startswith('0 '))

    def test_one_to_one_matches_and_empty_false_positives_count_frames(self):
        rows = [CaptureEntry(str(i), 'session', str(i), '', 320, 240, ('demo',)) for i in range(22)]
        frames = {name: [FramePredictions((BOX,), (Pick(BOX, .9), Pick(BOX, .8))),
                         FramePredictions((OTHER,), (Pick(BOX, .7),)),
                         FramePredictions((), (Pick(BOX, .8), Pick(OTHER, .6))),
                         *[FramePredictions((), ()) for _ in range(19)]] for name in CLASSES}
        scores = room.score(rows, frames, .75)
        self.assertEqual(scores['target_free'], {'frames': 20, 'false_positive_frames': 1, 'rate': .05})
        self.assertEqual(scores['classes']['book']['demo']['matched'], 1)
        self.assertEqual(scores['classes']['book']['precision'], 1 / 3)
        self.assertEqual(scores['classes']['book']['recall'], .5)
        self.assertAlmostEqual(scores['classes']['book']['ap50'], 51 / 101)
        self.assertEqual(scores['classes']['book']['representative_misses'][0]['frame_id'], '1')
        self.assertFalse(scores['passed'])

    def test_native_label_matches_at_exactly_half_iou(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            for index in range(2):
                capture = CaptureRun(root, f'session-{index}', ('demo',))
                image = np.full((240, 320, 3), index, dtype=np.uint8)
                frame = Frame(image, 1, 1, 0.0, datetime.now(timezone.utc))
                row = capture.save(frame, now=0.0)
                write_label(root, row, [(0, (0, 0, 20, 20))], classes=CLASSES)
            fold = room.load_fold(root)

        rows = [row for row, _, _ in fold.entries]
        frames = {name: [FramePredictions(tuple(targets[name]),
                                         (Pick((0, 0, 20, 10), .8),) if name == 'bottle' else ())
                         for _, _, targets in fold.entries] for name in CLASSES}
        self.assertEqual(room.score(rows, frames, .5)['classes']['bottle']['tp'], 2)

    def test_threshold_does_not_admit_two_false_positive_frames(self):
        rows = [CaptureEntry(str(i), 'session', str(i), '', 320, 240, ('demo',)) for i in range(22)]
        frames = {name: [FramePredictions((BOX,), (Pick(BOX, .9),)),
                         FramePredictions((BOX,), (Pick(BOX, .55),)),
                         FramePredictions((), (Pick(BOX, .6),)),
                         FramePredictions((), (Pick(BOX, .59),)),
                         *[FramePredictions((), ()) for _ in range(18)]] for name in CLASSES}
        self.assertEqual(room.select_confidence(rows, frames), .9)

    def test_validation_report_freezes_configuration_and_rejects_leakage(self):
        val = room.Fold([], {'fingerprint': 'val', 'sessions': ['val-1', 'val-2']}, {'a', 'b'})
        test = room.Fold([], {'fingerprint': 'test', 'sessions': ['test-1', 'test-2']}, {'c'})
        room.check_holdout(val, test)
        with self.assertRaisesRegex(DatasetError, 'duplicate'):
            room.check_holdout(val, room.Fold([], test.identity, {'b'}))
        with self.assertRaisesRegex(DatasetError, 'session'):
            room.check_holdout(val, room.Fold([], {'sessions': ['val-2']}, {'c'}))
        with TemporaryDirectory() as folder:
            path = Path(folder) / 'validation.json'
            path.write_text(json.dumps({'split': 'val', 'identity': {'imgsz': 320}, 'confidence': 0.54}))
            self.assertEqual(room.read_frozen(path, {'imgsz': 320})[0], 0.54)
            with self.assertRaisesRegex(DatasetError, 'differs'):
                room.read_frozen(path, {'imgsz': 640})

