"""Score one-class box predictions on the frozen XIAO study."""

from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass
from itertools import groupby

from .colab_training import CONFIDENCE, IMAGE_SIZE
from .dataset import read_image, read_label
from .detector import MAX_DETECTIONS, NMS_IOU, load_detector


Box = tuple[float, float, float, float]
IOU_THRESHOLDS = tuple(index / 100 for index in range(50, 100, 5))
SETTINGS = {'device': 'cpu', 'imgsz': IMAGE_SIZE, 'confidence_floor': CONFIDENCE,
            'nms_iou': NMS_IOU, 'agnostic_nms': False, 'rect': True, 'max_det': MAX_DETECTIONS,
            'metric_iou': 0.5, 'ap_iou_thresholds': list(IOU_THRESHOLDS), 'ap_recall_points': 101}


@dataclass(frozen=True)
class Pick:
    box: Box
    confidence: float


@dataclass(frozen=True)
class FramePredictions:
    targets: tuple[Box, ...]
    predictions: tuple[Pick, ...]
    tags: tuple[str, ...] = ()


def intersection_over_union(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union > 0 else 0.0


def ranked_matches(frames, iou):
    predictions = sorted(((pick, index) for index, frame in enumerate(frames) for pick in frame.predictions),
                         key=lambda value: value[0].confidence, reverse=True)
    unmatched = [dict(enumerate(frame.targets)) for frame in frames]
    result = []
    for pick, index in predictions:
        targets = unmatched[index]
        overlaps = [(intersection_over_union(pick.box, box), label) for label, box in targets.items()]
        overlap, label = max(overlaps, key=lambda value: value[0], default=(0.0, None))
        matched = overlap >= iou
        if matched:
            del targets[label]
        result.append((pick.confidence, matched, index))
    return result


def select_threshold(frames):
    label_count = sum(len(frame.targets) for frame in frames)
    tp = predictions = best_tp = 0
    best_denominator = label_count or 1
    threshold = 1.0
    # Include all equal scores together, because the display cutoff includes equality.
    for confidence, events in groupby(ranked_matches(frames, 0.5), key=lambda match: match[0]):
        for _, matched, _ in events:
            tp += int(matched)
            predictions += 1
        denominator = label_count + predictions
        if tp * best_denominator > best_tp * denominator:
            best_tp, best_denominator = tp, denominator
            threshold = confidence
    return threshold


def average_precision(matches, labels):
    if not matches or not labels:
        return 0.0
    tp = 0
    recall, precision = [], []
    for count, (_, matched, _) in enumerate(matches, start=1):
        tp += int(matched)
        recall.append(tp / labels)
        precision.append(tp / count)
    for index in range(len(precision) - 2, -1, -1):
        precision[index] = max(precision[index], precision[index + 1])
    precision.append(0.0)
    return sum(precision[bisect_left(recall, value / 100)] for value in range(101)) / 101


def measure(frames, threshold):
    labels = sum(len(frame.targets) for frame in frames)
    ap = []
    matches = ranked_matches(frames, 0.5)
    for iou in IOU_THRESHOLDS:
        ap.append(average_precision(matches if iou == 0.5 else ranked_matches(frames, iou), labels))
    selected = [(matched, index) for score, matched, index in matches if score >= threshold]
    hits = Counter(index for matched, index in selected if matched)
    tp = sum(hits.values())
    fp = len(selected) - tp
    empty_count = sum(not frame.targets for frame in frames)
    empty_fp = len({index for _, index in selected if not frames[index].targets})
    slice_counts = {}
    for index, frame in enumerate(frames):
        for tag in set(frame.tags):
            counts = slice_counts.setdefault(tag, Counter())
            counts.update(images=1, labels=len(frame.targets), misses=len(frame.targets) - hits[index])
    return {'ap50': ap[0], 'map50_95': sum(ap) / len(ap), 'threshold': threshold,
            'tp': tp, 'fp': fp, 'fn': labels - tp, 'images': len(frames), 'labels': labels,
            'precision': tp / len(selected) if selected else 0.0,
            'recall': tp / labels if labels else 0.0,
            'f1': 2 * tp / (labels + len(selected)) if labels + len(selected) else 0.0,
            'empty_frames': empty_count, 'empty_false_positives': empty_fp,
            'empty_false_positive_rate': empty_fp / empty_count if empty_count else None,
            'tagged_examples': slice_counts}


def promotion_checks(baseline, candidate):
    epsilon = 1e-12
    empty_baseline = baseline['empty_false_positive_rate']
    empty_candidate = candidate['empty_false_positive_rate']
    return {
        'map50_95_gain_at_least_0.02': candidate['map50_95'] - baseline['map50_95'] >= 0.02 - epsilon,
        'recall_loss_at_most_0.02': baseline['recall'] - candidate['recall'] <= 0.02 + epsilon,
        'empty_fp_increase_at_most_0.02': empty_baseline is not None and empty_candidate is not None
        and empty_candidate - empty_baseline <= 0.02 + epsilon,
    }


def predict_fold(study, checkpoint, split):
    detector = load_detector(checkpoint, confidence=CONFIDENCE, image_size=IMAGE_SIZE)
    frames = []
    for row, _ in study.entries:
        if row.split != split:
            continue
        image = read_image(row.image_path(study.root))
        labels = read_label(row.label_path(study.root))
        targets = tuple(((x - width / 2) * row.width, (y - height / 2) * row.height,
                         (x + width / 2) * row.width, (y + height / 2) * row.height)
                        for x, y, width, height in labels)
        predictions = []
        for (x1, y1, x2, y2), score in detector.predict_people(image):
            clipped = (max(0.0, x1), max(0.0, y1), min(row.width, x2), min(row.height, y2))
            if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                predictions.append(Pick(clipped, score))
        frames.append(FramePredictions(targets, tuple(predictions), row.tags))
    return frames
