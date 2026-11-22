import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from human_tracker.colab_training import (
    AUGMENTED_SETTINGS, DATASET_SHA256, SEED, digest, prepare_dataset, recipe,
    require_matching_batch, restore_recovery, verify_recipes,
)


class TrainingTests(unittest.TestCase):
    def test_recipes_vary_only_intended_training_transforms(self):
        plain = recipe('A')
        augmented = recipe('B')
        verify_recipes(plain, augmented)
        self.assertEqual({key for key in plain if plain[key] != augmented[key]}, set(AUGMENTED_SETTINGS))
        self.assertEqual(plain['seed'], SEED)
        self.assertEqual(plain['imgsz'], 320)
        self.assertEqual(plain['batch'], 8)
        self.assertEqual(plain['epochs'], 40)
        self.assertEqual(plain['patience'], 10)
        self.assertEqual(plain['augmentations'], [])
        with self.assertRaisesRegex(ValueError, 'batch'):
            verify_recipes(plain, recipe('B', batch=4))
        with self.assertRaisesRegex(ValueError, 'hsv_v'):
            verify_recipes(plain, {**augmented, 'hsv_v': 0.0})

    def test_archive_hash_prevents_extracting_wrong_data(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            archive = root / 'xiao-v1.zip'
            archive.write_bytes(b'wrong dataset')
            self.assertNotEqual(digest(archive), DATASET_SHA256)
            with self.assertRaisesRegex(ValueError, 'Dataset ZIP SHA-256'):
                prepare_dataset(archive, root / 'extracted')
            self.assertFalse((root / 'extracted').exists())

    def test_automatic_batch_reduction_blocks_the_inconsistent_run(self):
        require_matching_batch(SimpleNamespace(batch_size=8), 8)
        with self.assertRaisesRegex(RuntimeError, 'batch size 4'):
            require_matching_batch(SimpleNamespace(batch_size=4), 8)

    def test_recovery_uses_only_epochs_committed_with_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            run = Path(folder)
            recovery = run / 'recovery'
            recovery.mkdir()
            (recovery / 'best-epoch-1.pt').write_bytes(b'committed')
            (recovery / 'best-epoch-2.pt').write_bytes(b'uncommitted')
            (run / 'results.csv').write_text('epoch,loss\n1,1.2\n2,1.0\n')
            best = run / 'weights' / 'best.pt'
            restore_recovery(best, run / 'results.csv', recovery, 1, {'epoch': [1], 'loss': [1.2]})
            self.assertEqual(best.read_bytes(), b'committed')
            self.assertEqual((run / 'results.csv').read_text(), 'epoch,loss\n1,1.2\n')


if __name__ == '__main__':
    unittest.main()
