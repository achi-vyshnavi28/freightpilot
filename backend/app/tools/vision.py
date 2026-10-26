"""Yard camera pipeline: detect trucks, track them, read plates, measure dwell time.

The detector (YOLO) and tracker (ByteTrack) come from `ultralytics`, an optional
dependency (pip install ".[vision]"). Everything after detection is plain Python
so it is unit-tested without a GPU: plate cleanup, majority voting across
frames, gate-line crossing and dwell computation.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from ..validators import is_valid_vehicle, normalize_vehicle

TRUCK_CLASSES = {"truck", "bus"}  # COCO labels; Indian trucks are often detected as bus


@dataclass
class Detection:
    frame: int
    track_id: int
    label: str
    bbox: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    plate_text: str | None = None
    plate_conf: float = 0.0


@dataclass
class YardEvent:
    vehicle_no: str
    track_id: int
    entered_at: str | None
    exited_at: str | None
    dwell_minutes: float | None
    plate_confidence: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


_OCR_DIGIT = str.maketrans({"O": "0", "D": "0", "I": "1", "S": "5", "B": "8", "Z": "2"})
_OCR_ALPHA = str.maketrans({"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"})


def clean_plate(raw: str | None) -> str | None:
    """Repair a plate read by OCR. Indian plates are LL DD L{0,3} DDDD, so position tells us
    whether an ambiguous glyph should be a letter or a digit."""
    plate = normalize_vehicle(raw)
    if not plate or len(plate) < 6:
        return None
    if is_valid_vehicle(plate):
        return plate
    state = plate[:2].translate(_OCR_ALPHA)
    district = plate[2:4].translate(_OCR_DIGIT)
    tail = plate[-4:].translate(_OCR_DIGIT)
    series = plate[4:-4].translate(_OCR_ALPHA)
    fixed = state + district + series + tail
    return fixed if is_valid_vehicle(fixed) else None


@dataclass
class TrackState:
    first_frame: int
    last_frame: int
    plate_votes: Counter = field(default_factory=Counter)
    centers: list[tuple[int, float]] = field(default_factory=list)  # (frame, y center)


class DwellCalculator:
    """Turns per-frame tracked detections into entry/exit events.

    A truck enters when its box centre crosses `gate_y` moving down the frame and
    exits when it crosses back up (camera mounted over the gate, yard below).
    """

    def __init__(self, fps: float, start_time: datetime, gate_y: float) -> None:
        self.fps, self.start_time, self.gate_y = fps, start_time, gate_y
        self.tracks: dict[int, TrackState] = {}

    def update(self, det: Detection) -> None:
        if det.label not in TRUCK_CLASSES:
            return
        state = self.tracks.setdefault(det.track_id, TrackState(det.frame, det.frame))
        state.last_frame = det.frame
        state.centers.append((det.frame, (det.bbox[1] + det.bbox[3]) / 2))
        plate = clean_plate(det.plate_text)
        if plate:
            state.plate_votes[plate] += det.plate_conf or 0.5

    def _time(self, frame: int) -> str:
        return (self.start_time + timedelta(seconds=frame / self.fps)).isoformat()

    def events(self) -> list[YardEvent]:
        out = []
        for track_id, st in self.tracks.items():
            if not st.plate_votes:
                continue
            plate, votes = st.plate_votes.most_common(1)[0]
            total = sum(st.plate_votes.values())
            entered = exited = None
            for (_f0, y0), (f1, y1) in zip(st.centers, st.centers[1:], strict=False):
                if y0 < self.gate_y <= y1 and entered is None:
                    entered = f1
                elif y0 >= self.gate_y > y1 and entered is not None:
                    exited = f1
            dwell = (exited - entered) / self.fps / 60 if entered is not None and exited is not None else None
            out.append(
                YardEvent(
                    vehicle_no=plate,
                    track_id=track_id,
                    entered_at=self._time(entered) if entered is not None else None,
                    exited_at=self._time(exited) if exited is not None else None,
                    dwell_minutes=round(dwell, 1) if dwell is not None else None,
                    plate_confidence=round(votes / total, 2),
                )
            )
        return out


def run_video(path: str, start_time: datetime, gate_y_ratio: float = 0.5, weights: str = "yolo11n.pt") -> list[YardEvent]:
    """Full pipeline on a video file. Needs `pip install ".[vision]"`."""
    import cv2
    from ultralytics import YOLO

    model = YOLO(weights)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 720
    cap.release()
    calc = DwellCalculator(fps, start_time, gate_y=height * gate_y_ratio)
    for frame_idx, result in enumerate(model.track(path, stream=True, persist=True, tracker="bytetrack.yaml", verbose=False)):
        if result.boxes is None or result.boxes.id is None:
            continue
        for box, tid, cls in zip(
            result.boxes.xyxy.tolist(), result.boxes.id.int().tolist(), result.boxes.cls.int().tolist(), strict=True
        ):
            plate_text, plate_conf = read_plate(result.orig_img, box)
            calc.update(Detection(frame_idx, tid, model.names[cls], tuple(box), plate_text, plate_conf))
    return calc.events()


def read_plate(image, box) -> tuple[str | None, float]:
    """Crop the lower part of the truck box and OCR it. Swap in a fine-tuned plate model here."""
    try:
        import easyocr
    except ImportError:
        return None, 0.0
    x1, y1, x2, y2 = map(int, box)
    crop = image[y1 + (y2 - y1) // 2 : y2, x1:x2]
    reader = _easyocr_reader(easyocr)
    best = max(reader.readtext(crop), key=lambda r: r[2], default=None)
    return (best[1], float(best[2])) if best else (None, 0.0)


_reader = None


def _easyocr_reader(easyocr):
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


# Simulated events so the fleet agent has yard data without a camera.
_DEMO_EVENTS: dict[str, list[dict]] = defaultdict(list)


def record_yard_events(events: list[YardEvent]) -> None:
    for e in events:
        _DEMO_EVENTS[e.vehicle_no].append(e.to_dict())


def get_yard_events(vehicle_no: str) -> list[dict]:
    return list(_DEMO_EVENTS.get(normalize_vehicle(vehicle_no) or "", []))
