"""Command-line entry points for camera capture and dataset preparation."""

import argparse
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .annotation import label_frames, review_frames, window_open
from .capture import open_camera
from .dataset import CaptureRun, DEFAULT_ROOT, DatasetError, HEIGHT, SPLITS, WIDTH, export_dataset, live_frame, validate_dataset
from .detector import ROOM_CLASSES
from .detection_app import CaptureWorker


def capture_loop(cv, stream, run, *, interval=None, count=None, clock=time.monotonic):
    window = 'Capture XIAO images'
    next_due = None
    connection = None
    while count is None or len(run.rows) < count:
        state = stream.state
        if state.error:
            raise state.error
        now = clock()
        frame = state.frame if live_frame(state.frame, now) else None
        active_connection = frame.connection_id if frame is not None else None
        if active_connection != connection:
            connection = active_connection
            next_due = now + interval if connection is not None and interval is not None else None
        image = frame.image.copy() if frame is not None else np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        status = state.status if frame is not None or state.status != 'Streaming' else 'Waiting for a fresh frame'
        title = f'{len(run.rows)} saved | {status}'
        cv.putText(image, title[:46], (6, 17), cv.FONT_HERSHEY_SIMPLEX, 0.43, (80, 230, 130), 1, cv.LINE_AA)
        cv.imshow(window, image)
        key = cv.waitKey(10) & 0xFF
        if key in (ord('q'), ord('Q'), 27) or not window_open(cv, window):
            break
        now = clock()
        current = stream.state.frame
        if frame is None or not live_frame(current, now) or current.connection_id != frame.connection_id:
            next_due = None
            connection = None
            continue
        if key != 32 and (next_due is None or now < next_due):
            continue
        row = run.save(frame if key == 32 else current, now=now)
        if row is not None:
            print(f'Saved {len(run.rows)}  {row.filename}', flush=True)
            if interval is not None:
                next_due = now + interval
    return len(run.rows)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('Choose a positive number.')
    return number


def positive_seconds(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError('Choose a finite positive interval in seconds.')
    return number


def capture_main(arguments):
    parser = argparse.ArgumentParser(prog='human_tracker capture', description='Save native XIAO frames without overlays.')
    parser.add_argument('--port', required=True, help='XIAO USB serial port, for example COM7')
    parser.add_argument('--session', required=True, help='Recording session ID, reused only for the same session')
    parser.add_argument('--tag', action='append', default=[], help='Condition tag, repeatable')
    parser.add_argument('--output', type=Path, default=DEFAULT_ROOT, help='Capture folder (default datasets/xiao)')
    parser.add_argument('--timed', action='store_true', help='Save a new frame every 2 seconds while streaming')
    parser.add_argument('--interval', type=positive_seconds, help='Enable timed capture at this interval in seconds')
    parser.add_argument('--count', type=positive_int, help='Stop once this many frames have been saved')
    args = parser.parse_args(arguments)
    interval = args.interval or (2.0 if args.timed else None)
    try:
        run = CaptureRun(args.output, args.session, args.tag)
        print('Focus the preview. Space saves a frame. Q or Esc quits.', flush=True)
        if interval is not None:
            print(f'Timed capture is active every {interval:g} seconds while streaming.', flush=True)
        try:
            with CaptureWorker(open_camera(args.port)) as stream:
                captured = capture_loop(cv2, stream, run, count=args.count, interval=interval)
        finally:
            cv2.destroyAllWindows()
        print(f'Saved {captured} raw PNG(s) for session {args.session} in {args.output}.')
        return 1 if args.count is not None and captured < args.count else 0
    except (DatasetError, OSError, cv2.error) as exc:
        parser.exit(1, f'Capture error: {exc}\n')


def dataset_main(arguments):
    parser = argparse.ArgumentParser(prog='human_tracker dataset', description='Label, validate and freeze a YOLO dataset.')
    subparsers = parser.add_subparsers(dest='action', required=True)
    label = subparsers.add_parser('label', help='Draw or edit boxes in local captures')
    review = subparsers.add_parser('review', help='Review a sample of saved labels')
    validate = subparsers.add_parser('validate', help='Validate the capture folder or an extracted ZIP')
    export = subparsers.add_parser('export', help='Build a validated session-grouped YOLO ZIP')
    for command in (label, review, validate, export):
        command.add_argument('root', type=Path, nargs='?', default=DEFAULT_ROOT, help='Dataset folder (default datasets/xiao)')
    label.add_argument('--session', help='Only label frames from this session')
    label.add_argument('--edit', action='store_true', help='Include previously labeled frames')
    label.add_argument('--room-objects', action='store_true', help='Label bottle, book and cup')
    review.add_argument('--sample', type=positive_int, default=30, help='Maximum frames to show (default 30)')
    review.add_argument('--tag', help='Only review frames with this condition tag')
    review.add_argument('--split', choices=SPLITS, help='Only review this split of an extracted dataset')
    export.add_argument('--output', type=Path, required=True, help='New output ZIP path')
    export.add_argument('--test-session', action='append', default=[], help='Session reserved for test, repeat for each')
    args = parser.parse_args(arguments)
    try:
        if args.action == 'label':
            label_frames(args.root, session=args.session, edit=args.edit,
                         classes=ROOM_CLASSES if args.room_objects else ())
        elif args.action == 'review':
            review_frames(args.root, count=args.sample, tag=args.tag, split=args.split)
        elif args.action == 'validate':
            entries = validate_dataset(args.root)
            positive = sum(is_positive for _, is_positive in entries)
            print(f'PASS: {len(entries)} labeled PNGs, {positive} positive, {len(entries) - positive} explicit negatives.')
        else:
            if args.output.suffix.lower() != '.zip':
                parser.error('--output must end in .zip')
            report = export_dataset(args.root, args.output, test_sessions=args.test_session)
            print(f'PASS: {args.output}')
            print(', '.join(f'{split}={report.counts[split]}' for split in SPLITS))
            print(f'Manifest SHA-256 {report.manifest_sha256}')
            print(f'Archive SHA-256  {report.archive_sha256}')
    except (DatasetError, OSError, cv2.error, UnicodeError) as exc:
        parser.exit(1, f'Dataset error: {exc}\n')
    return 0
