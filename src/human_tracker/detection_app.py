"""Live and sequential replay entry points for object detection."""

import argparse
from collections import Counter
from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

from .capture import Frame, open_camera
from .detector import ROOM_CLASSES, load_detector


@dataclass(frozen=True)
class CaptureState:
    frame: Frame | None = None
    decoded_count: int = 0
    status: str = "Waiting for USB camera"
    error: Exception | None = None


class CaptureWorker:
    def __init__(self, camera):
        self.camera = camera
        self.state = CaptureState()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.read, name="xiao-capture", daemon=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.stop.set()
        self.thread.join(timeout=1)

    def read(self):
        try:
            while not self.stop.is_set():
                self.camera.poll()
                self.state = CaptureState(self.camera.latest, self.camera.decoded_count, self.camera.status)
                self.stop.wait(0.005)
        except Exception as exc:
            self.state = CaptureState(error=exc)
        finally:
            self.camera.close()


def current_result(processed, latest):
    return latest is not None and processed.connection_id == latest.connection_id


def draw_boxes(cv2, image, detections):
    canvas = image.copy()
    for detection in detections:
        left, top, right, bottom = detection.box
        cv2.rectangle(canvas, (left, top), (right, bottom), (60, 220, 90), 2)
        cv2.putText(canvas, f"{detection.class_name} {detection.confidence:.2f}", (left, max(12, top - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.43, (60, 220, 90), 1, cv2.LINE_AA)
    return canvas


def preview(cv2, image, title):
    window = "XIAO detection"
    cv2.putText(image, title[:58], (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (235, 235, 235), 1)
    cv2.imshow(window, image)
    key = cv2.waitKey(1) & 0xFF
    try:
        visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
    except cv2.error:
        visible = False
    return visible and key not in (ord("q"), 27)


def live(cv2, np, detector, args):
    camera = open_camera(args.source.partition(":")[2])
    blank = np.zeros((240, 320, 3), dtype=np.uint8)
    detector.detect(blank)
    boot = time.monotonic()
    start = None
    initial_decoded = 0
    initial_connection = None
    interrupted = False
    last_sequence = 0
    last_status = None
    drawn = None
    processed = 0
    latency = []
    with CaptureWorker(camera) as worker:
        while True:
            state = worker.state
            if state.error is not None:
                raise RuntimeError(f"USB capture failed: {state.error}")
            frame, status = state.frame, state.status
            if status != last_status:
                print(f"t={time.monotonic() - boot:.2f}s {status}", flush=True)
                last_status = status

            if frame is None:
                drawn = None
                if start is not None:
                    interrupted = True
                if start is None and args.duration and time.monotonic() - boot >= 15:
                    raise RuntimeError("No XIAO frames after 15 seconds. Check the port and firmware.")
            elif frame.sequence != last_sequence:
                last_sequence = frame.sequence
                if start is None:
                    start = time.monotonic()
                    initial_decoded = frame.sequence - 1
                    initial_connection = frame.connection_id
                interrupted |= frame.connection_id != initial_connection
                image = draw_boxes(cv2, frame.image, detector.detect(frame.image))
                latest = worker.state.frame
                if current_result(frame, latest):
                    drawn = (frame, image)
                    if not args.headless:
                        fps = processed / max(time.monotonic() - start, 0.001)
                        if not preview(cv2, image.copy(), f"{status} | {fps:.1f} processed FPS"):
                            return int(args.duration is not None)
                    processed += 1
                    if args.duration is not None:
                        latency.append((time.monotonic() - frame.received_at) * 1000)
            else:
                if not args.headless:
                    image = drawn[1] if drawn and current_result(drawn[0], frame) else blank
                    fps = processed / max(time.monotonic() - start, 0.001)
                    if not preview(cv2, image.copy(), f"{status} | {fps:.1f} processed FPS"):
                        return int(args.duration is not None)

            if not args.headless and frame is None:
                if not preview(cv2, blank.copy(), status):
                    return int(args.duration is not None)

            now = time.monotonic()
            if start is not None and args.duration and now - start >= args.duration:
                elapsed = now - start
                decoded = worker.state.decoded_count - initial_decoded
                p95 = sorted(latency)[math.ceil(processed * 0.95) - 1] if processed else math.inf
                interrupted |= not current_result(frame, worker.state.frame) if frame is not None else True
                passed = processed / elapsed >= 5 and p95 <= 350 and not interrupted
                print(f"{'PASS' if passed else 'FAIL'}: {elapsed:.2f}s, "
                      f"{decoded / elapsed:.2f} decoded FPS, {processed / elapsed:.2f} processed FPS, "
                      f"{decoded - processed} skipped, p95 {p95:.1f}ms "
                      f"({'headless overlay' if args.headless else 'preview'}), interrupted={interrupted}")
                return int(not passed)

            time.sleep(0.005)


def replay_frames(cv2, path):
    if not path.is_file():
        raise ValueError(f"Replay file not found: {path}")
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Could not decode image: {path}")
        yield image
        return
    reader = cv2.VideoCapture(str(path))
    try:
        if not reader.isOpened():
            raise ValueError(f"Could not open video: {path}")
        while True:
            ok, image = reader.read()
            if not ok:
                break
            yield image
    finally:
        reader.release()


def replay(cv2, detector, args):
    last = None
    frames = 0
    counts = Counter()
    classes = ROOM_CLASSES if args.room_objects else ("person",)
    with_frames = replay_frames(cv2, Path(args.source))
    try:
        for image in with_frames:
            detections = detector.detect(image)
            last = draw_boxes(cv2, image, detections)
            frames += 1
            frame_counts = Counter(detection.class_name for detection in detections)
            counts.update(frame_counts)
            title = " | ".join(f"{name} {frame_counts[name]}" for name in classes)
            if not args.headless and not preview(cv2, last.copy(), title):
                break
    finally:
        with_frames.close()
    if last is None:
        raise ValueError("The replay contained no decodable frames.")
    if args.output and not cv2.imwrite(args.output, last):
        raise ValueError(f"Could not write {args.output}")
    totals = ", ".join(f"{counts[name]} {name} boxes" for name in classes)
    print(f"Replay: {frames} frames, {totals}.")
    return 0


def main(mode, arguments):
    parser = argparse.ArgumentParser(prog=f"python -m human_tracker {mode}")
    parser.add_argument("--source", required=True, help="serial:COM8" if mode == "live" else "Local image or video")
    parser.add_argument("--model", default="models/yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--room-objects", action="store_true", help="Detect bottle, book and cup")
    if mode == "live":
        parser.add_argument("--duration", type=float, help="Benchmark for this many seconds after warm-up")
    else:
        parser.add_argument("--output", help="Save the last annotated frame to an image")
    args = parser.parse_args(arguments)
    if not 0 < args.confidence <= 1 or args.imgsz <= 0 or args.imgsz % 32:
        parser.error("Confidence must be in (0, 1] and --imgsz must be a positive multiple of 32.")
    if mode == "live":
        prefix, _, port = args.source.partition(":")
        if prefix != "serial" or not port:
            parser.error("Use a serial source such as serial:COM8.")
        if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
            parser.error("--duration must be positive.")

    try:
        import cv2
        import numpy as np

        detector = load_detector(args.model, confidence=args.confidence, image_size=args.imgsz,
                                 room_objects=args.room_objects)
        return live(cv2, np, detector, args) if mode == "live" else replay(cv2, detector, args)
    except ImportError as exc:
        parser.exit(2, f"Missing dependency ({exc}). Install the project with the detection extra.\n")
    except (ValueError, OSError, RuntimeError) as exc:
        parser.exit(2, f"{exc}\n")
    except KeyboardInterrupt:
        return 130
    finally:
        if not args.headless and "cv2" in locals():
            cv2.destroyAllWindows()
