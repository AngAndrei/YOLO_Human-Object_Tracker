"""Run selected-class detection with a local Ultralytics checkpoint."""

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np


NMS_IOU = 0.7
MAX_DETECTIONS = 300
ROOM_CLASSES = ("bottle", "book", "cup")


@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    confidence: float
    class_name: str = "person"


class ObjectDetector:
    def __init__(self, model, *, classes, confidence=0.35, image_size=320):
        if model.task != "detect":
            raise ValueError("Select a YOLO object detection checkpoint.")
        names = model.names
        entries = list(names.items() if isinstance(names, dict) else enumerate(names))
        self.class_names = {}
        for name in classes:
            matches = [int(class_id) for class_id, label in entries if label.casefold() == name]
            if len(matches) != 1:
                raise ValueError(f"The checkpoint must contain exactly one '{name}' class.")
            self.class_names[matches[0]] = name
        self.model = model
        self.confidence = confidence
        # Ultralytics uses a strict cutoff; retain the display boundary for our inclusive filter.
        self.prediction_cutoff = float(np.nextafter(np.float32(confidence), np.float32(0.0)))
        self.image_size = image_size

    def predict_objects(self, image):
        result = self.model.predict(
            source=image, device="cpu", imgsz=self.image_size,
            classes=list(self.class_names), conf=self.prediction_cutoff, iou=NMS_IOU,
            max_det=MAX_DETECTIONS, agnostic_nms=False, rect=True, verbose=False,
        )[0]
        for x1, y1, x2, y2, score, class_id in result.boxes.data.tolist():
            class_name = self.class_names.get(class_id)
            if class_name is None or score < self.confidence:
                continue
            if not all(math.isfinite(value) for value in (x1, y1, x2, y2, score)):
                continue
            yield (x1, y1, x2, y2), score, class_name

    def detect(self, image):
        height, width = image.shape[:2]
        detections = []
        for (x1, y1, x2, y2), score, class_name in self.predict_objects(image):
            left = max(0, min(round(x1), width - 1))
            right = max(0, min(round(x2), width - 1))
            top = max(0, min(round(y1), height - 1))
            bottom = max(0, min(round(y2), height - 1))
            if right > left and bottom > top:
                detections.append(Detection((left, top, right, bottom), score, class_name))
        return detections


class PersonDetector(ObjectDetector):
    def __init__(self, model, *, confidence=0.35, image_size=320):
        super().__init__(model, classes=("person",), confidence=confidence, image_size=image_size)
        self.person_id = next(iter(self.class_names))

    def predict_people(self, image):
        for box, score, _ in self.predict_objects(image):
            yield box, score


def load_detector(checkpoint, *, confidence=0.35, image_size=320, room_objects=False):
    path = Path(checkpoint)
    if path.suffix.lower() != ".pt" or not path.is_file():
        raise ValueError(f"Local YOLO checkpoint not found: {path}. Place official yolo11n.pt there.")
    from ultralytics import YOLO

    model = YOLO(str(path))
    if room_objects:
        return ObjectDetector(model, classes=ROOM_CLASSES, confidence=confidence, image_size=image_size)
    return PersonDetector(model, confidence=confidence, image_size=image_size)
