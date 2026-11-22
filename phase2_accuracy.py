"""Measure person detections and empty-scene false positives on live XIAO frames."""

import argparse
import sys
import time

from human_tracker.capture import open_camera
from human_tracker.detection_app import CaptureWorker, draw_boxes
from human_tracker.detector import load_detector


SAMPLES_PER_SCENE = 10


def wait_for_frame(worker, *, previous=None, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = worker.state
        if state.error is not None:
            raise RuntimeError(f"Camera read failed: {state.error}") from state.error
        frame = state.frame
        if frame is not None:
            if previous is None:
                return frame
            if frame.connection_id != previous.connection_id:
                raise RuntimeError("The camera connection changed during capture. Restart the check.")
            if frame.sequence > previous.sequence:
                return frame
        time.sleep(0.02)
    raise RuntimeError(f"Timed out waiting for a fresh frame. Camera status: {worker.state.status}")


def capture_scene(worker, instruction):
    input(f"{instruction}\nPress Enter for a five-second countdown. ")
    time.sleep(5)
    previous = worker.state.frame
    samples = []
    for index in range(SAMPLES_PER_SCENE):
        if index:
            time.sleep(1)
        previous = wait_for_frame(worker, previous=previous)
        samples.append(previous.image.copy())
        print(f"Captured {index + 1}/{SAMPLES_PER_SCENE}", end="\r", flush=True)
    print()
    return samples


def score_accuracy(detect, occupied, empty, review):
    if len(occupied) != SAMPLES_PER_SCENE or len(empty) != SAMPLES_PER_SCENE:
        raise ValueError("The check needs 10 person frames and 10 empty frames.")

    valid_frames = 0
    for index, image in enumerate(occupied, start=1):
        detections = detect(image)
        confirmed = review(index, image, detections)
        valid_frames += int(bool(detections) and confirmed)
    false_positive_frames = sum(bool(detect(image)) for image in empty)
    return valid_frames, false_positive_frames


def review_person(cv2, index, image, detections):
    window = "XIAO accuracy review"
    drawn = draw_boxes(cv2, image, detections)
    cv2.putText(drawn, f"{index}/10  Y = valid box  N = miss  Q = quit", (6, 234),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.imshow(window, drawn)

    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("y"), ord("Y"), ord("n"), ord("N")):
            return key in (ord("y"), ord("Y"))
        try:
            visible = cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            visible = False
        if key in (ord("q"), ord("Q"), 27) or not visible:
            raise KeyboardInterrupt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="COM7", help="XIAO runtime serial port")
    parser.add_argument("--model", default="models/yolo11n.pt", help="Local person checkpoint")
    args = parser.parse_args(argv)

    try:
        import cv2

        detector = load_detector(args.model)
        with CaptureWorker(open_camera(args.port)) as worker:
            wait_for_frame(worker, timeout=15)
            occupied = capture_scene(worker, "Step into view and vary your pose or distance.")
            empty = capture_scene(worker, "Step out of view and leave the background in frame.")

        print("Review the person frames. Press Y if a box encloses a visible person, otherwise N.")
        cv2.namedWindow("XIAO accuracy review", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("XIAO accuracy review", 640, 480)
        valid, false_positives = score_accuracy(
            detector.detect, occupied, empty,
            lambda index, image, detections: review_person(cv2, index, image, detections),
        )
        passed = valid >= 8 and false_positives <= 1
        print(f"{'PASS' if passed else 'FAIL'}: valid person boxes in {valid}/10 person frames, "
              f"false positives in {false_positives}/10 empty frames.")
        return int(not passed)
    except (OSError, RuntimeError, ValueError, ImportError, EOFError) as error:
        print(error, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Accuracy check cancelled.", file=sys.stderr)
        return 130
    finally:
        if "cv2" in locals():
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
