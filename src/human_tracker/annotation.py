"""Draw boxes on local captures and inspect annotations."""

from collections import defaultdict, deque
import random

import cv2

from .dataset import (
    DatasetError, HEIGHT, WIDTH, exclude_frame, label_pixels, read_image, read_label, read_manifest, validate_dataset,
    write_label,
)


SCALE = 2


class BoxEditor:
    def __init__(self, boxes=()):
        self.boxes = list(boxes)
        self.class_id = None
        self.start = None
        self.end = None

    def mouse(self, event, x, y, _flags, _parameter):
        point = (max(0, min(WIDTH, (x + SCALE // 2) // SCALE)),
                 max(0, min(HEIGHT, (y + SCALE // 2) // SCALE)))
        if event == cv2.EVENT_LBUTTONDOWN and 0 <= y < HEIGHT * SCALE:
            self.start = self.end = point
        elif self.start is not None and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP):
            self.end = point
            if event == cv2.EVENT_LBUTTONUP:
                left, right = sorted((self.start[0], self.end[0]))
                top, bottom = sorted((self.start[1], self.end[1]))
                box = (left, top, right, bottom)
                record = box if self.class_id is None else (self.class_id, box)
                if left < right and top < bottom and record not in self.boxes:
                    self.boxes.append(record)
                self.start = self.end = None


def window_open(cv, window):
    try:
        return cv.getWindowProperty(window, cv.WND_PROP_VISIBLE) >= 1
    except cv.error:
        return False


def review_sample(entries, count):
    buckets = defaultdict(list)
    for row, positive in entries:
        buckets[row.split, positive].append((row, positive))
    rng = random.Random(42)
    queues = []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        queues.append(deque(bucket))
    chosen = []
    while queues and len(chosen) < count:
        for queue in queues:
            chosen.append(queue.popleft())
            if len(chosen) == count:
                break
        queues = [queue for queue in queues if queue]
    return chosen


def edit_window(root, rows, *, review=False, classes=()):
    room = bool(classes)
    window = 'Label XIAO room objects' if room else 'Label XIAO people'
    if review:
        window = 'XIAO annotation review'
    if not rows:
        print('No matching frames.')
        return 0
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)
    completed = 0
    active_class = 0
    class_keys = '    '.join(f'{index} {name}' for index, name in enumerate(classes, start=1))
    try:
        for index, row in enumerate(rows, start=1):
            label_path = row.label_path(root, room=room)
            labels = read_label(label_path, classes=classes) if label_path.is_file() else []
            editor = BoxEditor((class_id, label_pixels(box)) for class_id, box in labels) if room else BoxEditor(
                label_pixels(box) for box in labels)
            if room:
                editor.class_id = active_class
            source = cv2.resize(read_image(row.image_path(root)), (SCALE * WIDTH, SCALE * HEIGHT))
            if not review:
                cv2.setMouseCallback(window, editor.mouse)
            controls = ('Space or Enter next    Q quit' if review else
                        'Drag boxes    Enter save    E empty    U undo    S skip    X exclude    Q quit')
            detail = ', '.join(row.tags)
            while True:
                canvas = cv2.copyMakeBorder(source, 0, 72, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30))
                for record in editor.boxes:
                    class_id, (left, top, right, bottom) = record if room else (None, record)
                    cv2.rectangle(canvas, (left * SCALE, top * SCALE),
                                  (min(WIDTH * SCALE - 1, right * SCALE), min(HEIGHT * SCALE - 1, bottom * SCALE)),
                                  (80, 230, 130), 2)
                    if room:
                        cv2.putText(canvas, classes[class_id], (left * SCALE, max(13, top * SCALE - 4)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (80, 230, 130), 1, cv2.LINE_AA)
                if editor.start is not None:
                    cv2.rectangle(canvas, tuple(c * SCALE for c in editor.start),
                                  tuple(c * SCALE for c in editor.end), (80, 210, 245), 1)
                status = f'{index}/{len(rows)}  {row.split} {row.session_id}  |  {len(editor.boxes)} box(es)'
                if room:
                    status += f' | Drawing {classes[active_class]} '
                    detail = f'{class_keys}    {", ".join(row.tags)}'
                if row.exclusion_path(root, room=room).is_file():
                    status += ' | Excluded'
                for offset, line in enumerate((status, detail, controls), start=1):
                    cv2.putText(canvas, line[:90], (8, HEIGHT * SCALE + 20 * offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1, cv2.LINE_AA)
                cv2.imshow(window, canvas)
                key = cv2.waitKey(25) & 0xFF
                if key in (ord('q'), ord('Q'), 27) or not window_open(cv2, window):
                    return completed
                if review:
                    if key in (10, 13, 32):
                        completed += 1
                        break
                elif room and ord('1') <= key < ord('1') + len(classes):
                    active_class = editor.class_id = key - ord('1')
                elif key in (ord('u'), ord('U')) and editor.boxes:
                    editor.boxes.pop()
                elif key in (ord('s'), ord('S')):
                    break
                elif key in (ord('x'), ord('X')):
                    exclude_frame(root, row, room=room)
                    break
                elif (key in (10, 13) and editor.boxes) or (key in (ord('e'), ord('E')) and not editor.boxes):
                    write_label(root, row, editor.boxes, classes=classes)
                    completed += 1
                    break
    finally:
        cv2.destroyAllWindows()
    return completed


def label_frames(root, *, session=None, edit=False, classes=()):
    if (root / 'manifest.csv').is_file():
        raise DatasetError('Label in the original capture folder, then export a new ZIP.')
    room = bool(classes)
    rows = [row for row in read_manifest(root) if (session is None or row.session_id == session)
            and (edit or (not row.label_path(root, room=room).is_file()
                          and not row.exclusion_path(root, room=room).is_file()))]
    print(f'Saved labels on {edit_window(root, rows, classes=classes)} frame(s).')


def review_frames(root, *, count, tag=None, split=None):
    entries = [(row, positive) for row, positive in validate_dataset(root)
               if (tag is None or tag in row.tags) and (split is None or split == row.split)]
    frames = review_sample(entries, count)
    print(f'Reviewed {edit_window(root, [row for row, _ in frames], review=True)} of {len(frames)} sampled frames.')
