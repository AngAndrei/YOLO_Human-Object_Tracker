"""Freeze a shared room-object confidence cutoff and score independent XIAO captures."""

import argparse
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from itertools import groupby
import json
import math
from pathlib import Path
from platform import python_version
from tempfile import TemporaryDirectory

from .colab_training import CONFIDENCE, digest, save_json
from .dataset import DatasetError, label_pixels, manifest_bytes, publish_new, read_image, read_label, read_manifest
from .detector import MAX_DETECTIONS, NMS_IOU, ROOM_CLASSES, load_detector
from .evaluation import FramePredictions, Pick, average_precision, ranked_matches


IOU = 0.50
MIN_INSTANCES = 20
MIN_RECALL = 0.80
MAX_EMPTY_RATE = 0.05
DEMO_TAG = 'demo'


@dataclass(frozen=True)
class Fold:
    entries: list
    identity: dict
    image_hashes: set


def load_fold(root):
    if (root / 'manifest.csv').is_file():
        raise DatasetError('Use an original room capture folder rather than a frozen person study.')
    rows = [row for row in read_manifest(root) if not row.exclusion_path(root, room=True).exists()]
    sessions = sorted({row.session_id for row in rows})
    if len(sessions) < 2:
        raise DatasetError(f'{root} needs at least two independent capture sessions.')
    fingerprint = sha256(manifest_bytes(rows))
    image_hashes = set()
    entries = []
    for row in rows:
        image = read_image(row.image_path(root))
        image_hash = sha256(image.tobytes()).hexdigest()
        if image_hash in image_hashes:
            raise DatasetError(f'{root} contains a duplicate frame: {row.filename}.')
        image_hashes.add(image_hash)
        labels = read_label(row.label_path(root, room=True), classes=ROOM_CLASSES)
        fingerprint.update(bytes.fromhex(image_hash))
        fingerprint.update(json.dumps(labels, separators=(',', ':')).encode('ascii'))
        targets = {name: [] for name in ROOM_CLASSES}
        for class_id, box in labels:
            targets[ROOM_CLASSES[class_id]].append(label_pixels(box))
        entries.append((row, image, targets))
    return Fold(entries, {'fingerprint': fingerprint.hexdigest(), 'sessions': sessions}, image_hashes)


def check_holdout(validation, test):
    if set(validation.identity['sessions']) & set(test.identity['sessions']):
        raise DatasetError('A capture session appears in both validation and test.')
    if validation.image_hashes & test.image_hashes:
        raise DatasetError('A duplicate image appears in validation and test.')


def coverage(fold):
    return {
        'frames': len(fold.entries),
        'target_free_frames': sum(not any(targets.values()) for _, _, targets in fold.entries),
        'demo_instances': {
            name: sum(len(targets[name]) for row, _, targets in fold.entries if DEMO_TAG in row.tags)
            for name in ROOM_CLASSES
        },
        'tags': dict(Counter(tag for row, _, _ in fold.entries for tag in set(row.tags))),
    }


def collect_predictions(fold, detector):
    frames = {name: [] for name in ROOM_CLASSES}
    for row, image, targets in fold.entries:
        picks = {name: [] for name in ROOM_CLASSES}
        for result in detector.detect(image):
            picks[result.class_name].append(Pick(result.box, result.confidence))
        for name in ROOM_CLASSES:
            frames[name].append(FramePredictions(tuple(targets[name]), tuple(picks[name]), row.tags))
    return frames


def empty_indices(frames):
    return {index for index, items in enumerate(zip(*(frames[name] for name in ROOM_CLASSES), strict=True))
            if not any(frame.targets for frame in items)}


def select_confidence(rows, frames):
    demo = {index for index, row in enumerate(rows) if DEMO_TAG in row.tags}
    empty = empty_indices(frames)
    demo_counts = {name: sum(len(frames[name][index].targets) for index in demo) for name in ROOM_CLASSES}
    if not empty or not all(demo_counts.values()):
        raise DatasetError('Validation needs target-free frames and at least one demo instance of each class.')
    total_labels = sum(len(frame.targets) for items in frames.values() for frame in items)
    events = [(confidence, matched, index, name) for name in ROOM_CLASSES
              for confidence, matched, index in ranked_matches(frames[name], IOU)]
    events.sort(key=lambda event: event[0], reverse=True)
    empty_hits = set()
    demo_hits = Counter()
    tp = predictions = 0
    best = (0, 0, 1.0)
    for confidence, group in groupby(events, key=lambda event: event[0]):
        for _, matched, index, name in group:
            predictions += 1
            if matched:
                tp += 1
                if index in demo:
                    demo_hits[name] += 1
            if index in empty:
                empty_hits.add(index)
        if len(empty_hits) / len(empty) > MAX_EMPTY_RATE:
            break
        worst_demo_recall = min(demo_hits[name] / demo_counts[name] for name in ROOM_CLASSES)
        best = max(best, (worst_demo_recall, 2 * tp / (total_labels + predictions), confidence))
    return best[2]


def score(rows, frames, confidence):
    empty = empty_indices(frames)
    hits_on_empty = set()
    classes = {}
    for name in ROOM_CLASSES:
        items = frames[name]
        matches = ranked_matches(items, IOU)
        labels = sum(len(frame.targets) for frame in items)
        selected = [(matched, index) for value, matched, index in matches if value >= confidence]
        hits = Counter(index for matched, index in selected if matched)
        hits_on_empty.update(index for _, index in selected if index in empty)
        tp = sum(hits.values())
        demo_labels = demo_hits = 0
        misses = []
        for index, (row, frame) in enumerate(zip(rows, items, strict=True)):
            if DEMO_TAG in row.tags:
                demo_labels += len(frame.targets)
                demo_hits += hits[index]
            if len(frame.targets) > hits[index] and len(misses) < 5:
                misses.append({'frame_id': row.frame_id, 'image': row.filename, 'session': row.session_id,
                               'tags': row.tags, 'labeled': len(frame.targets), 'matched': hits[index]})
        demo_recall = demo_hits / demo_labels if demo_labels else 0.0
        classes[name] = {
            'precision': tp / len(selected) if selected else 0.0,
            'recall': tp / labels if labels else 0.0,
            'ap50': average_precision(matches, labels),
            'tp': tp, 'fp': len(selected) - tp, 'fn': labels - tp,
            'demo': {'instances': demo_labels, 'matched': demo_hits, 'recall': demo_recall,
                     'passed': demo_labels >= MIN_INSTANCES and demo_recall >= MIN_RECALL},
            'representative_misses': misses,
        }
    rate = len(hits_on_empty) / len(empty) if empty else None
    return {
        'classes': classes,
        'target_free': {'frames': len(empty), 'false_positive_frames': len(hits_on_empty), 'rate': rate},
        'passed': (len(empty) >= MIN_INSTANCES and rate <= MAX_EMPTY_RATE
                   and all(stats['demo']['passed'] for stats in classes.values())),
    }


def read_frozen(path, identity):
    if not path.is_file():
        raise DatasetError('Run room-evaluate validate to freeze confidence before testing.')
    content = path.read_bytes()
    report = json.loads(content)
    if not isinstance(report, dict) or report.get('split') != 'val' or report.get('identity') != identity:
        raise DatasetError('Current validation data or inference configuration differs from the frozen report.')
    confidence = report.get('confidence')
    if type(confidence) not in (int, float) or not CONFIDENCE <= confidence <= 1:
        raise DatasetError('Frozen validation confidence is invalid.')
    return confidence, sha256(content).hexdigest()


def run(args):
    if args.mode == 'inspect':
        fold = load_fold(args.data_root / args.split)
        print(json.dumps({**fold.identity, **coverage(fold)}, indent=2))
        return 0

    split = 'val' if args.mode == 'validate' else 'test'
    destination = args.output / ('validation.json' if split == 'val' else 'test.json')
    if destination.exists():
        raise DatasetError(f'{destination} already exists. Use a new output directory for another run.')
    val = load_fold(args.data_root / 'val')
    identity = {
        'validation': val.identity,
        'checkpoint_sha256': digest(args.model),
        'settings': {'classes': list(ROOM_CLASSES), 'device': 'cpu', 'imgsz': args.imgsz, 'rect': True,
                     'nms_iou': NMS_IOU, 'agnostic_nms': False, 'max_det': MAX_DETECTIONS,
                     'confidence_floor': CONFIDENCE, 'match_iou': IOU, 'min_instances': MIN_INSTANCES,
                     'min_demo_recall': MIN_RECALL, 'max_empty_frame_rate': MAX_EMPTY_RATE},
        'runtime': {'python': python_version(), **{name: version(name) for name in
                     ('numpy', 'opencv-python', 'torch', 'ultralytics')}},
    }
    if split == 'test':
        confidence, validation_hash = read_frozen(args.output / 'validation.json', identity)
        fold = load_fold(args.data_root / 'test')
        check_holdout(val, fold)
    else:
        fold = val

    detector = load_detector(args.model, confidence=CONFIDENCE, image_size=args.imgsz, room_objects=True)
    rows = [row for row, _, _ in fold.entries]
    frames = collect_predictions(fold, detector)
    if split == 'val':
        confidence = select_confidence(rows, frames)
    report = {'split': split, 'identity': identity, 'dataset': fold.identity, 'confidence': confidence,
              'coverage': coverage(fold), **score(rows, frames, confidence)}
    if split == 'test':
        report['validation_sha256'] = validation_hash
    if digest(args.model) != identity['checkpoint_sha256']:
        raise DatasetError('The checkpoint changed during scoring. Retry with a stable model file.')
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output) as staging:
        path = Path(staging) / destination.name
        save_json(path, report)
        publish_new(path, destination)
    print(f'{split}: confidence={confidence!r} | {"PASS" if report["passed"] else "BELOW TARGET"} | {destination}')
    return int(split == 'test' and not report['passed'])


def main(arguments):
    parser = argparse.ArgumentParser(prog='human_tracker room-evaluate', description=__doc__)
    parser.add_argument('mode', choices=('inspect', 'validate', 'test'))
    parser.add_argument('--data-root', type=Path, default=Path('datasets/xiao-room'))
    parser.add_argument('--split', choices=('val', 'test'), default='val', help='Capture split to inspect')
    parser.add_argument('--output', type=Path, default=Path('build/phase8'))
    parser.add_argument('--model', type=Path, default=Path('models/yolo11n.pt'))
    parser.add_argument('--imgsz', type=int, default=320)
    args = parser.parse_args(arguments)
    if args.imgsz <= 0 or args.imgsz % 32:
        parser.error('--imgsz must be a positive multiple of 32.')
    try:
        return run(args)
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(2, f'Room evaluation error: {exc}\n')

