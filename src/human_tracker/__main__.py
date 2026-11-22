"""Run the Phase 1 camera preview or USB throughput check."""

import argparse
import sys
import time

from .capture import open_camera


def main():
    if len(sys.argv) > 1 and sys.argv[1] == 'room-evaluate':
        from .room_evaluation import main as evaluate_room

        return evaluate_room(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        from .evaluation_app import main as evaluate

        return evaluate(sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] in ("live", "replay"):
        from .detection_app import main as detect

        return detect(sys.argv[1], sys.argv[2:])
    if len(sys.argv) > 1 and sys.argv[1] in ('capture', 'dataset'):
        from .collection_app import capture_main, dataset_main

        command = capture_main if sys.argv[1] == 'capture' else dataset_main
        return command(sys.argv[2:])
    parser = argparse.ArgumentParser(description="Preview JPEG frames from the XIAO USB camera.")
    parser.epilog = "Use live or replay for detection, capture and dataset for data, and room-evaluate for room-object scoring."
    parser.add_argument("--port", required=True, help="Windows COM port, for example COM8")
    parser.add_argument("--headless", action="store_true", help="Display connection status in the console")
    parser.add_argument("--duration", type=float, help="Measure for this many seconds once the first frame arrives")
    parser.add_argument("--min-fps", type=float, default=8, help="Required average decoded FPS for a timed run")
    parser.add_argument("--snapshot", help="Save the first decoded camera frame")
    args = parser.parse_args()
    if args.duration is not None and args.duration <= 0:
        parser.error("--duration must be positive")

    try:
        import cv2
        import numpy as np
        camera = open_camera(args.port)
    except ImportError as exc:
        parser.exit(2, f"Missing dependency ({exc}). Install this project first.\n")

    start = None
    first_sequence = 0
    connection_id = None
    interrupted = False
    shapes = set()
    last_status = None
    last_sequence = 0
    deadline = time.monotonic() + 15
    try:
        while True:
            camera.poll()
            now = time.monotonic()
            frame = camera.latest
            if camera.status != last_status:
                print(f"{args.port}: {camera.status}", flush=True)
                last_status = camera.status
            if frame is not None and frame.sequence != last_sequence:
                last_sequence = frame.sequence
                if start is None:
                    start = frame.received_at
                    first_sequence = frame.sequence
                    connection_id = frame.connection_id
                    if args.snapshot and not cv2.imwrite(args.snapshot, frame.image):
                        raise RuntimeError(f"Could not save {args.snapshot}")
                shapes.add(frame.image.shape[:2])
            if start is not None and (frame is None or frame.connection_id != connection_id):
                interrupted = True
            if not args.headless:
                image = frame.image.copy() if frame is not None else np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(image, camera.status[:42], (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 230, 130), 1)
                cv2.imshow("XIAO USB camera", image)
                key = cv2.waitKey(1) & 0xFF
                try:
                    visible = cv2.getWindowProperty("XIAO USB camera", cv2.WND_PROP_VISIBLE) >= 1
                except cv2.error:
                    visible = False
                if key in (ord("q"), 27) or not visible:
                    return 0 if args.duration is None else 1
            if args.duration is not None:
                if start is None and now >= deadline:
                    print("FAIL: No decoded frame within 15 seconds.", file=sys.stderr)
                    return 1
                if start is not None and now - start >= args.duration:
                    count = camera.decoded_count - first_sequence
                    fps = count / (now - start)
                    sizes = ", ".join(f"{w}x{h}" for h, w in sorted(shapes))
                    passed = fps >= args.min_fps and shapes == {(240, 320)} and not interrupted
                    print(f"{'PASS' if passed else 'FAIL'}: {count} decoded frames in {now - start:.2f} s, "
                          f"{fps:.2f} FPS, {sizes}, disconnect={interrupted}")
                    return 0 if passed else 1
            time.sleep(0.01)
    except KeyboardInterrupt:
        return 130 if args.duration is not None else 0
    finally:
        camera.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())
