"""Store native XIAO frames and build a validated YOLO dataset."""

from collections import Counter, defaultdict
import csv
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import io
from itertools import product
import json
import math
import os
from pathlib import Path
import re
import tempfile
from uuid import uuid4
import zipfile

import cv2
import numpy as np


DEFAULT_ROOT = Path('datasets/xiao')
WIDTH, HEIGHT = 320, 240
MAX_FRAME_AGE = 0.5
SPLITS = ('train', 'val', 'test')
MIN_SESSIONS = {'train': 1, 'val': 2, 'test': 2}
FRACTIONS = {'train': 0.7, 'val': 0.15, 'test': 0.15}
FIELDS = ('frame_id', 'session_id', 'filename', 'received_utc', 'width', 'height', 'tags')
DATA_YAML = 'train: images/train\nval: images/val\ntest: images/test\nnames:\n  0: person\n'
SESSION_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_-]{0,63}')
FRAME_ID_PATTERN = re.compile(r'[0-9a-f]{32}')
PNG_SIGNATURE = b'\x89PNG\r\n\x1a\n'


class DatasetError(ValueError):
    pass


@dataclass(frozen=True)
class CaptureEntry:
    frame_id: str
    session_id: str
    filename: str
    received_utc: str
    width: int
    height: int
    tags: tuple[str, ...]
    split: str = ''

    def image_path(self, root):
        return root / 'images' / self.split / self.filename

    def label_path(self, root, *, room=False):
        return root / ('room-labels' if room else 'labels') / self.split / f'{self.frame_id}.txt'

    def exclusion_path(self, root, *, room=False):
        return root / ('room-excluded' if room else 'excluded') / f'{self.frame_id}.txt'


def write_atomic(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        incomplete = Path(stream.name)
    try:
        incomplete.write_bytes(content)
        os.replace(incomplete, path)
    finally:
        incomplete.unlink(missing_ok=True)


def manifest_bytes(rows, *, frozen=False):
    stream = io.StringIO(newline='')
    writer = csv.writer(stream, lineterminator='\n')
    writer.writerow((*FIELDS, 'split') if frozen else FIELDS)
    for row in rows:
        values = (row.frame_id, row.session_id, row.filename, row.received_utc,
                  row.width, row.height, json.dumps(row.tags, ensure_ascii=False))
        writer.writerow((*values, row.split) if frozen else values)
    return stream.getvalue().encode('utf-8')


def live_frame(frame, now):
    return frame is not None and 0 <= now - frame.received_at <= MAX_FRAME_AGE


class CaptureRun:
    def __init__(self, root, session_id, tags=()):
        if (root / 'manifest.csv').exists():
            raise DatasetError('Capture into the original collection folder or choose a new folder.')
        if not SESSION_PATTERN.fullmatch(session_id):
            raise DatasetError('Session ID must be 1 to 64 letters, digits, hyphens or underscores and start with a letter or digit.')
        if any(not tag.strip() for tag in tags):
            raise DatasetError('Condition tags must contain text.')
        self.root = root
        self.session_id = session_id
        self.tags = tuple(tags)
        self.manifest = root / 'manifests' / f'{uuid4().hex}.csv'
        self.rows = []
        self.seen = set()
        self.last_sequence = -1

    def save(self, frame, *, now):
        if not live_frame(frame, now) or frame.sequence <= self.last_sequence:
            return None
        if frame.received_utc is None or frame.image.shape != (HEIGHT, WIDTH, 3) or frame.image.dtype != np.uint8:
            raise DatasetError('Expected a timestamped native 320x240 color frame.')
        digest = hashlib.sha256(frame.image.tobytes()).digest()
        if digest in self.seen:
            return None
        frame_id = uuid4().hex
        row = CaptureEntry(frame_id, self.session_id, f'{frame_id}.png',
                           frame.received_utc.astimezone(timezone.utc).isoformat(), WIDTH, HEIGHT, self.tags)
        encoded, png = cv2.imencode('.png', frame.image)
        if not encoded:
            raise DatasetError('Could not encode the camera frame as PNG.')
        image_path = row.image_path(self.root)
        write_atomic(image_path, png.tobytes())
        try:
            write_atomic(self.manifest, manifest_bytes([*self.rows, row]))
        except OSError:
            image_path.unlink(missing_ok=True)
            raise
        self.rows.append(row)
        self.seen.add(digest)
        self.last_sequence = frame.sequence
        return row


def read_manifest(root):
    frozen = (root / 'manifest.csv').is_file()
    paths = [root / 'manifest.csv'] if frozen else sorted((root / 'manifests').glob('*.csv'))
    rows = []
    seen = set()
    fields = (*FIELDS, 'split') if frozen else FIELDS
    for path in paths:
        try:
            with path.open(encoding='utf-8', newline='') as stream:
                reader = csv.reader(stream, strict=True)
                if tuple(next(reader, ())) != fields:
                    raise DatasetError(f'Invalid header in {path}')
                for values in reader:
                    try:
                        raw = dict(zip(fields, values, strict=True))
                        tags = json.loads(raw['tags'])
                        received = datetime.fromisoformat(raw['received_utc'])
                        frame_id = raw['frame_id']
                        width, height = int(raw['width']), int(raw['height'])
                        split = raw.get('split', '')
                        if (not FRAME_ID_PATTERN.fullmatch(frame_id) or raw['filename'] != f'{frame_id}.png'
                                or not SESSION_PATTERN.fullmatch(raw['session_id'])
                                or received.utcoffset() != timezone.utc.utcoffset(None)
                                or (width, height) != (WIDTH, HEIGHT) or (frozen and split not in SPLITS)
                                or not isinstance(tags, list)
                                or any(not isinstance(tag, str) or not tag.strip() for tag in tags)):
                            raise ValueError('Invalid frame metadata')
                    except ValueError as exc:
                        raise DatasetError(f'Invalid manifest row at {path}:{reader.line_num}') from exc
                    if frame_id in seen:
                        raise DatasetError(f'Duplicate frame ID {frame_id} in {path}')
                    seen.add(frame_id)
                    rows.append(CaptureEntry(frame_id, raw['session_id'], raw['filename'], raw['received_utc'],
                                             width, height, tuple(tags), split))
        except (csv.Error, UnicodeError) as exc:
            raise DatasetError(f'Cannot read capture manifest {path}') from exc
    if not rows:
        raise DatasetError(f'No captured frames in {root}')
    return sorted(rows, key=lambda row: (datetime.fromisoformat(row.received_utc), row.frame_id))


def read_image(path):
    if not path.is_file():
        raise DatasetError(f'Missing image {path}')
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise DatasetError(f'Invalid PNG {path}')
    try:
        image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    except cv2.error as exc:
        raise DatasetError(f'Invalid PNG {path}') from exc
    if image is None or image.shape != (HEIGHT, WIDTH, 3) or image.dtype != np.uint8:
        raise DatasetError(f'Invalid PNG {path}. Expected a 320x240 color image.')
    return image


def label_pixels(box):
    x, y, width, height = box
    return (round((x - width / 2) * WIDTH), round((y - height / 2) * HEIGHT),
            round((x + width / 2) * WIDTH), round((y + height / 2) * HEIGHT))


def write_label(root, row, boxes, *, classes=()):
    lines = []
    seen = set()
    for record in boxes:
        class_id, (left, top, right, bottom) = record if classes else (0, record)
        if type(class_id) is not int or class_id not in range(len(classes) or 1):
            raise DatasetError('Unknown box class.')
        if not (0 <= left < right <= row.width and 0 <= top < bottom <= row.height):
            raise DatasetError('A label box must have positive area and be inside the frame.')
        values = ((left + right) / (2 * row.width), (top + bottom) / (2 * row.height),
                  (right - left) / row.width, (bottom - top) / row.height)
        key = class_id, values
        if key in seen:
            raise DatasetError('Each box within a class must be distinct.')
        seen.add(key)
        lines.append(f'{class_id} ' + ' '.join(format(value, '.12g') for value in values) + '\n')
    write_atomic(row.label_path(root, room=bool(classes)), ''.join(lines).encode('ascii'))
    row.exclusion_path(root, room=bool(classes)).unlink(missing_ok=True)


def exclude_frame(root, row, *, room=False):
    write_atomic(row.exclusion_path(root, room=room), b'')


def read_label(path, *, classes=()):
    if not path.is_file():
        raise DatasetError(f'Missing label {path}. Confirm empty frames explicitly.')
    boxes = []
    seen = set()
    valid_ids = {str(index) for index in range(len(classes) or 1)}
    try:
        for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), start=1):
            fields = line.split()
            if not fields:
                continue
            try:
                class_id, *coordinates = fields
                x, y, w, h = values = tuple(map(float, coordinates))
                tolerance = 1e-9
                key = class_id, values
                if (class_id not in valid_ids or not all(math.isfinite(value) and 0 <= value <= 1 for value in values)
                        or w <= 0 or h <= 0 or key in seen
                        or min(x - w / 2, y - h / 2) < -tolerance
                        or max(x + w / 2, y + h / 2) > 1 + tolerance):
                    raise ValueError('Invalid class or box')
            except ValueError as exc:
                raise DatasetError(f'Invalid YOLO label at {path}:{number}') from exc
            boxes.append((int(class_id), values) if classes else values)
            seen.add(key)
    except UnicodeError as exc:
        raise DatasetError(f'Invalid YOLO label encoding in {path}') from exc
    return boxes


def check_split_coverage(entries):
    counts = defaultdict(set)
    kinds = defaultdict(set)
    assignments = {}
    for row, positive in entries:
        previous = assignments.setdefault(row.session_id, row.split)
        if previous != row.split:
            raise DatasetError(f'Session {row.session_id} is present in more than one split.')
        counts[row.split].add(row.session_id)
        kinds[row.split].add(positive)
    for split in SPLITS:
        if len(counts[split]) < MIN_SESSIONS[split] or kinds[split] != {True, False}:
            raise DatasetError(f'{split} needs at least {MIN_SESSIONS[split]} distinct session(s) and both positive and negative frames.')


def validate_dataset(root):
    frozen = (root / 'manifest.csv').is_file()
    if frozen:
        manifest_hash = hashlib.sha256((root / 'manifest.csv').read_bytes()).hexdigest()
        checksum = root / 'manifest.sha256'
        if not checksum.is_file() or checksum.read_text(encoding='ascii').strip() != manifest_hash:
            raise DatasetError('The manifest SHA-256 does not match manifest.csv.')
        yaml = root / 'data.yaml'
        if not yaml.is_file() or yaml.read_text(encoding='utf-8') != DATA_YAML:
            raise DatasetError('data.yaml must define class 0 and the three dataset image paths.')
    entries = []
    expected_images = set()
    expected_labels = set()
    digests = {}
    for row in read_manifest(root):
        image_path, label_path = row.image_path(root), row.label_path(root)
        expected_images.add(image_path)
        expected_labels.add(label_path)
        if not frozen and row.exclusion_path(root).is_file():
            continue
        digest = hashlib.sha256(read_image(image_path).tobytes()).digest()
        if digest in digests:
            raise DatasetError(f'Duplicate image pixels in {image_path} and {digests[digest]}.')
        digests[digest] = image_path
        entries.append((row, bool(read_label(label_path))))
    for folder, pattern, expected in (('images', '*', expected_images), ('labels', '*.txt', expected_labels)):
        unexpected = sorted(path for path in (root / folder).rglob(pattern) if path not in expected and path.is_file())
        if unexpected:
            raise DatasetError(f'File without a matching manifest row: {unexpected[0]}')
    if not entries:
        raise DatasetError('No included frames. Label frames to include them in the dataset.')
    if frozen:
        check_split_coverage(entries)
    return entries


def assign_splits(entries, test_sessions):
    counts = Counter(row.session_id for row, _ in entries)
    kinds = defaultdict(set)
    for row, positive in entries:
        kinds[row.session_id].add(positive)
    if len(counts) < sum(MIN_SESSIONS.values()):
        raise DatasetError('Collect at least five independent sessions before exporting.')

    reserved = set(test_sessions)
    if reserved and (len(reserved) < MIN_SESSIONS['test'] or not reserved <= counts.keys()
                     or set().union(*(kinds[session] for session in reserved)) != {True, False}):
        raise DatasetError('Reserved test sessions must include at least two existing sessions and both positive and negative frames.')
    movable = tuple(split for split in SPLITS if split != 'test' or not reserved)
    available = sorted(counts.keys() - reserved, key=lambda session: (-counts[session], session))
    targets = {split: FRACTIONS[split] * len(entries) for split in SPLITS}
    best = None
    best_cost = math.inf
    for use_mixed in product((True, False), repeat=len(movable)):
        remaining = available.copy()
        assignments = dict.fromkeys(reserved, 'test')
        size = Counter({'test': sum(counts[session] for session in reserved)})
        seeds = []
        for split, mixed in zip(movable, use_mixed):
            required = ({True, False},) if mixed else ({True}, {False})
            seeds.extend((split, kind) for kind in required)
        seeded_counts = Counter(assignments.values())
        pending_seeds = Counter(split for split, _ in seeds)
        for split, kind in seeds:
            candidates = [session for session in remaining if kinds[session] == kind]
            if not candidates:
                break
            needed = max(pending_seeds[split], MIN_SESSIONS[split] - seeded_counts[split])
            target = (targets[split] - size[split]) / needed
            chosen = min(candidates, key=lambda session: abs(target - counts[session]))
            remaining.remove(chosen)
            assignments[chosen] = split
            size[split] += counts[chosen]
            seeded_counts[split] += 1
            pending_seeds[split] -= 1
        else:
            extras = [split for split in movable for _ in range(MIN_SESSIONS[split] - seeded_counts[split])]
            if len(remaining) < len(extras):
                continue
            for split in extras:
                chosen = min(remaining, key=lambda session: abs(targets[split] - size[split] - counts[session]))
                remaining.remove(chosen)
                assignments[chosen] = split
                size[split] += counts[chosen]
            for session in remaining:
                split = max(movable, key=lambda split: targets[split] - size[split])
                assignments[session] = split
                size[split] += counts[session]
            cost = sum((size[split] - targets[split]) ** 2 for split in SPLITS)
            if cost < best_cost:
                best, best_cost = assignments, cost
    if best is None:
        raise DatasetError('Cannot give all splits independent sessions and both positive and negative frames. Collect more sessions with both kinds of frames.')
    return best


@dataclass(frozen=True)
class ExportReport:
    counts: dict[str, int]
    manifest_sha256: str
    archive_sha256: str


def publish_new(source, destination):
    # Windows rename and POSIX hard-link both refuse to replace an existing destination.
    if os.name == 'nt':
        source.rename(destination)
    else:
        os.link(source, destination)


def export_dataset(root, output, *, test_sessions=()):
    if (root / 'manifest.csv').exists():
        raise DatasetError('Export from the original capture folder.')
    checksum = output.with_suffix(output.suffix + '.sha256')
    if output.exists() or checksum.exists():
        raise DatasetError(f'Choose a new ZIP name. {output} or its checksum already exists.')
    entries = validate_dataset(root)
    assignments = assign_splits(entries, test_sessions)
    frozen = [(replace(row, split=assignments[row.session_id]), positive) for row, positive in entries]
    check_split_coverage(frozen)
    manifest = manifest_bytes([row for row, _ in frozen], frozen=True)
    manifest_hash = hashlib.sha256(manifest).hexdigest()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as directory:
        staging = Path(directory)
        archive = staging / 'dataset.zip'
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as stream:
            stream.writestr('data.yaml', DATA_YAML)
            stream.writestr('manifest.csv', manifest)
            stream.writestr('manifest.sha256', manifest_hash + '\n')
            for (raw, _), (row, _) in zip(entries, frozen):
                stream.write(raw.image_path(root), row.image_path(Path()).as_posix())
                stream.write(raw.label_path(root), row.label_path(Path()).as_posix())
        extracted = staging / 'extracted'
        with zipfile.ZipFile(archive) as stream:
            stream.extractall(extracted)
        validate_dataset(extracted)
        with archive.open('rb') as stream:
            archive_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
        staged_checksum = staging / 'dataset.sha256'
        write_atomic(staged_checksum, f'{archive_hash}  {output.name}\n'.encode('utf-8'))
        publish_new(staged_checksum, checksum)
        try:
            publish_new(archive, output)
        except OSError:
            checksum.unlink(missing_ok=True)
            raise
    counts = Counter(row.split for row, _ in frozen)
    return ExportReport(dict(counts), manifest_hash, archive_hash)
