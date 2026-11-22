from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import cv2
import numpy as np

from human_tracker.capture import Frame
from human_tracker.dataset import (
    CaptureRun,
    DatasetError,
    export_dataset,
    read_manifest,
    validate_dataset,
    write_label,
)


RECEIVED = datetime(2026, 11, 20, 1, 2, 3, tzinfo=timezone.utc)


def frame(n, *, sequence=None, connection=1, received_at=10.0):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    image[40:60, 80:100, 0] = n
    return Frame(image, n if sequence is None else sequence, connection, received_at, RECEIVED)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.root = self.directory / 'captures'

    def saved(self, n, session='room', boxes=()):
        row = CaptureRun(self.root, session).save(frame(n), now=10.1)
        if boxes is not None:
            write_label(self.root, row, boxes)
        return row

    def archive(self):
        for n in range(5):
            session = f'visit-{n}'
            self.saved(n * 2 + 1, session, [(120, 60, 200, 180)])
            self.saved(n * 2 + 2, session)
        path = self.directory / 'frozen.zip'
        report = export_dataset(self.root, path, test_sessions=('visit-3', 'visit-4'))
        extracted = self.directory / 'unpacked'
        with zipfile.ZipFile(path) as archive:
            self.assertIsNone(archive.testzip())
            archive.extractall(extracted)
        return path, report, extracted

    def test_capture_keeps_raw_pixels_timestamps_and_ids_across_restarts(self):
        first = CaptureRun(self.root, 'room', ('side-light', 'shadow-only'))
        sample = frame(7)
        row = first.save(sample, now=10.1)
        self.assertEqual((row.width, row.height), (320, 240))
        self.assertEqual(row.session_id, 'room')
        self.assertEqual(row.received_utc, RECEIVED.isoformat())
        self.assertEqual(row.tags, ('side-light', 'shadow-only'))
        image = cv2.imread(str(self.root / 'images' / row.filename))
        np.testing.assert_array_equal(sample.image, image)
        another = CaptureRun(self.root, 'room').save(frame(8, sequence=7), now=10.3)
        self.assertNotEqual(another.frame_id, row.frame_id)
        self.assertCountEqual(read_manifest(self.root), [row, another])

    def test_skips_disconnect_stale_frame_reused_sequence_and_identical_pixels(self):
        run = CaptureRun(self.root, 'room')
        self.assertIsNone(run.save(None, now=10.1))
        self.assertIsNone(run.save(frame(1, received_at=8), now=10.1))
        self.assertIsNotNone(run.save(frame(2), now=10.1))
        self.assertIsNone(run.save(frame(3, sequence=2), now=10.2))
        self.assertIsNone(run.save(frame(2, sequence=3, connection=2), now=10.2))
        self.assertIsNotNone(run.save(frame(4, sequence=4, connection=2), now=10.3))
        self.assertEqual(len(read_manifest(self.root)), 2)

    def test_manifest_failure_does_not_leave_unlisted_png(self):
        run = CaptureRun(self.root, 'room')
        replace = os.replace

        def fail_manifest(source, destination):
            if destination.suffix == '.csv':
                raise OSError('disk error')
            replace(source, destination)

        with patch('human_tracker.dataset.os.replace', side_effect=fail_manifest):
            with self.assertRaisesRegex(OSError, 'disk error'):
                run.save(frame(3), now=10.1)
        self.assertEqual(list((self.root / 'images').iterdir()), [])
        self.assertIsNotNone(run.save(frame(3), now=10.1))

    def test_local_annotation_emits_normalized_boxes_and_explicit_empty_labels(self):
        row = self.saved(1, boxes=[(120, 60, 200, 180), (0, 0, 320, 240)])
        label = self.root / 'labels' / f'{row.frame_id}.txt'
        self.assertEqual(label.read_text().splitlines(), ['0 0.5 0.5 0.25 0.5', '0 0.5 0.5 1 1'])
        empty = self.saved(2)
        self.assertEqual((self.root / 'labels' / f'{empty.frame_id}.txt').read_bytes(), b'')
        self.assertEqual(len(validate_dataset(self.root)), 2)

    def test_validation_rejects_missing_label_and_corrupt_png(self):
        row = self.saved(1, boxes=None)
        with self.assertRaisesRegex(DatasetError, 'Missing label'):
            validate_dataset(self.root)
        write_label(self.root, row, [])
        (self.root / 'images' / row.filename).write_bytes(b'corrupt png')
        with self.assertRaisesRegex(DatasetError, 'Invalid PNG'):
            validate_dataset(self.root)

    def test_validation_rejects_bad_classes_coordinates_and_duplicate_boxes(self):
        row = self.saved(1)
        label = self.root / 'labels' / f'{row.frame_id}.txt'
        for invalid in ('1 .5 .5 .3 .3', '0 .5 .5 .3 0', '0 .1 .5 .3 .3',
                        '0 NaN .5 .3 .3', '0 1.01 .5 .3 .3', '0 .5 .5 .3',
                        '0 .5 .5 .3 .3\n0 0.5 .5 .3 .3'):
            with self.subTest(invalid=invalid):
                label.write_text(invalid)
                with self.assertRaisesRegex(DatasetError, 'Invalid YOLO label'):
                    validate_dataset(self.root)

    def test_duplicate_pixels_are_rejected_even_if_png_encoding_differs(self):
        first = self.saved(1)
        duplicate = self.saved(1, session='different')
        self.assertTrue(cv2.imwrite(str(self.root / 'images' / duplicate.filename), frame(1).image,
                                    [cv2.IMWRITE_PNG_COMPRESSION, 9]))
        self.assertNotEqual((self.root / 'images' / first.filename).read_bytes(),
                            (self.root / 'images' / duplicate.filename).read_bytes())
        with self.assertRaisesRegex(DatasetError, 'Duplicate image'):
            validate_dataset(self.root)

    def test_export_contains_paired_data_with_independent_holdout_sessions(self):
        path, report, extracted = self.archive()
        rows = validate_dataset(extracted)
        self.assertEqual(Counter(row.split for row, _ in rows), {'train': 2, 'val': 4, 'test': 4})
        self.assertEqual({row.session_id for row, _ in rows if row.split == 'test'}, {'visit-3', 'visit-4'})
        self.assertEqual(report.archive_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(report.manifest_sha256, hashlib.sha256((extracted / 'manifest.csv').read_bytes()).hexdigest())
        yaml = (extracted / 'data.yaml').read_text()
        for split in ('train', 'val', 'test'):
            self.assertIn(f'{split}: images/{split}', yaml)
            self.assertEqual(len(list((extracted / 'images' / split).glob('*.png'))), report.counts[split])
            self.assertEqual(len(list((extracted / 'labels' / split).glob('*.txt'))), report.counts[split])
        self.assertIn('0: person', yaml)

    def test_validator_detects_session_leakage_and_checks_manifest_hash(self):
        _, _, extracted = self.archive()
        manifest = extracted / 'manifest.csv'
        with manifest.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            fields = reader.fieldnames
            rows = list(reader)
        train = next(row for row in rows if row['split'] == 'train')
        next(row for row in rows if row['split'] == 'val')['session_id'] = train['session_id']
        with manifest.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with self.assertRaisesRegex(DatasetError, 'manifest SHA-256'):
            validate_dataset(extracted)
        (extracted / 'manifest.sha256').write_text(hashlib.sha256(manifest.read_bytes()).hexdigest() + '\n')
        with self.assertRaisesRegex(DatasetError, 'more than one split'):
            validate_dataset(extracted)

    def test_validator_rejects_unmanifested_jpeg(self):
        _, _, extracted = self.archive()
        self.assertTrue(cv2.imwrite(str(extracted / 'images' / 'train' / 'extra.jpg'), frame(200).image))
        with self.assertRaisesRegex(DatasetError, 'without a matching manifest row'):
            validate_dataset(extracted)

    def test_checksum_write_failure_leaves_no_published_archive_and_allows_retry(self):
        for n in range(5):
            self.saved(n * 2 + 1, f'visit-{n}', [(120, 60, 200, 180)])
            self.saved(n * 2 + 2, f'visit-{n}')
        path = self.directory / 'frozen.zip'
        with patch('human_tracker.dataset.write_atomic', side_effect=OSError('checksum write failed')):
            with self.assertRaisesRegex(OSError, 'checksum write failed'):
                export_dataset(self.root, path)
        self.assertFalse(path.exists())
        report = export_dataset(self.root, path)
        self.assertEqual(path.with_suffix('.zip.sha256').read_text().split()[0], report.archive_sha256)

    def test_explicit_exclusion_allows_skipping_a_frame_and_exporting_the_rest(self):
        from human_tracker.dataset import exclude_frame

        for n in range(5):
            self.saved(n * 2 + 1, f'visit-{n}', [(120, 60, 200, 180)])
            self.saved(n * 2 + 2, f'visit-{n}')
        skipped = self.saved(50, 'visit-0', boxes=None)
        with self.assertRaisesRegex(DatasetError, 'Missing label'):
            validate_dataset(self.root)
        exclude_frame(self.root, skipped)
        self.assertTrue(skipped.image_path(self.root).is_file())
        path = self.directory / 'frozen.zip'
        report = export_dataset(self.root, path)
        self.assertEqual(sum(report.counts.values()), 10)
        with zipfile.ZipFile(path) as archive:
            self.assertNotIn(skipped.frame_id.encode(), archive.read('manifest.csv'))
        write_label(self.root, skipped, [])
        self.assertEqual(len(validate_dataset(self.root)), 11)

    def test_split_uses_positive_only_and_negative_only_sessions(self):
        for n in range(6):
            self.saved(n + 1, f'visit-{n}', [(10, 10, 20, 20)] if n % 2 == 0 else [])
        report = export_dataset(self.root, self.directory / 'frozen.zip')
        self.assertEqual(report.counts, {'train': 2, 'val': 2, 'test': 2})

    def test_split_refuses_insufficient_sessions_or_unbalanced_reserved_test(self):
        self.saved(1, 'one', [(0, 0, 320, 240)])
        self.saved(2, 'two')
        self.saved(3, 'three')
        path = self.directory / 'frozen.zip'
        with self.assertRaisesRegex(DatasetError, 'at least five'):
            export_dataset(self.root, path)
        self.saved(4, 'four')
        self.saved(5, 'five')
        with self.assertRaisesRegex(DatasetError, 'Reserved test sessions'):
            export_dataset(self.root, path, test_sessions=('two', 'three'))


if __name__ == '__main__':
    unittest.main()
