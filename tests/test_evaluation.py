import unittest

from human_tracker.evaluation import (
    FramePredictions, Pick, measure, promotion_checks, select_threshold,
)


PERSON = (0, 0, 10, 10)
OTHER = (20, 20, 30, 30)


class EvaluationTests(unittest.TestCase):
    def test_duplicate_miss_and_empty_frame_are_counted_once(self):
        frames = [FramePredictions((PERSON, OTHER), (Pick(PERSON, 0.9), Pick(PERSON, 0.8))),
                  FramePredictions((), (Pick(PERSON, 0.7),))]
        score = measure(frames, 0.7)
        self.assertEqual((score['tp'], score['fp'], score['fn']), (1, 2, 1))
        self.assertEqual(score['precision'], 1 / 3)
        self.assertEqual(score['recall'], 0.5)
        self.assertEqual(score['empty_false_positives'], 1)
        self.assertEqual(score['empty_frames'], 1)
        self.assertAlmostEqual(score['ap50'], 51 / 101)
        self.assertAlmostEqual(score['map50_95'], 51 / 101)

    def test_threshold_uses_higher_confidence_when_f1_is_tied(self):
        frames = [FramePredictions((PERSON, OTHER), (Pick(PERSON, 0.9), Pick(PERSON, 0.8),
                                                      Pick(PERSON, 0.7), Pick(OTHER, 0.6)))]
        self.assertEqual(select_threshold(frames), 0.9)
        tied = [FramePredictions((PERSON, OTHER), (Pick(PERSON, 0.9), Pick(OTHER, 0.9)))]
        self.assertEqual(select_threshold(tied), 0.9)

    def test_ap_accounts_for_box_localization_at_higher_iou(self):
        frames = [FramePredictions((PERSON,), (Pick((0, 0, 10, 6), 0.9),))]
        score = measure(frames, 0.9)
        self.assertEqual(score['ap50'], 1.0)
        self.assertAlmostEqual(score['map50_95'], 0.3)

    def test_promotion_gates_use_absolute_differences(self):
        baseline = {'map50_95': 0.7, 'recall': 0.9, 'empty_false_positive_rate': 0.1}
        candidate = {'map50_95': 0.72, 'recall': 0.88, 'empty_false_positive_rate': 0.12}
        self.assertTrue(all(promotion_checks(baseline, candidate).values()))
        candidate['empty_false_positive_rate'] = 0.121
        self.assertFalse(all(promotion_checks(baseline, candidate).values()))


if __name__ == '__main__':
    unittest.main()
