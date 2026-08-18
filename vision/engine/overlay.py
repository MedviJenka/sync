from collections.abc import Iterable
from dataclasses import dataclass
from typing import Optional

NEON_CYAN = (255, 255, 0)
NEON_MAGENTA = (255, 0, 255)
NEON_BLUE = (255, 160, 0)


@dataclass(frozen=True)
class Point:
    x: int
    y: int


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError("width must be greater than 0")
        if self.height <= 0:
            raise ValueError("height must be greater than 0")

    @property
    def right(self) -> int:
        return self.x + self.width

    @property
    def bottom(self) -> int:
        return self.y + self.height

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.width // 2, y=self.y + self.height // 2)

    def point_at(self, x_percent: int, y_percent: int) -> Point:
        return Point(
            x=self.x + (self.width * x_percent) // 100,
            y=self.y + (self.height * y_percent) // 100,
        )


@dataclass(frozen=True)
class DetectedFace:
    face: Rect
    eyes: tuple[Rect, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "eyes", tuple(self.eyes))


@dataclass(frozen=True)
class DetectedHand:
    palm: Rect
    fingers_up: tuple[int, int, int, int, int] = (0, 0, 0, 0, 0)
    landmarks: tuple[Point, ...] = ()
    handedness: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "landmarks", tuple(self.landmarks))
        fingers = tuple(int(value) for value in self.fingers_up)
        if len(fingers) != 5:
            raise ValueError("fingers_up must contain exactly five values")
        if any(value not in (0, 1) for value in fingers):
            raise ValueError("fingers_up values must be 0 or 1")
        object.__setattr__(self, "fingers_up", fingers)


@dataclass(frozen=True)
class OverlayPrimitive:

    key:           str
    points:        tuple[Point, ...]
    color:         tuple[int, int, int] = NEON_CYAN
    thickness:     int                  = 2
    accent_color:  Optional[tuple[int]] = None
    corner_length: int                  = 0
    label:         Optional[str]        = None
    radius:        int                  = 4
    font_scale:    float                = 0.7

    def __post_init__(self) -> None:
        object.__setattr__(self, "points", tuple(self.points))
        if self.radius < 0:
            raise ValueError("radius must be greater than or equal to 0")
        if self.font_scale <= 0:
            raise ValueError("font_scale must be greater than 0")


def build_biometrics_overlay(detection: DetectedFace) -> tuple[OverlayPrimitive, ...]:
    face = detection.face
    corner_length = max(12, min(face.width, face.height) // 7)
    label_anchor = Point(face.x, max(12, face.y - 6))
    baseline_margin = max(corner_length, face.width // 4)
    baseline_y = face.bottom + 6

    return (
        OverlayPrimitive(
            key="face_hud_box",
            points=(Point(face.x, face.y), Point(face.right, face.bottom)),
            color=NEON_CYAN,
            thickness=2,
            accent_color=NEON_BLUE,
            corner_length=corner_length,
        ),
        OverlayPrimitive(
            key="face_label",
            points=(label_anchor,),
            color=NEON_CYAN,
            thickness=1,
            label="TRACK",
            font_scale=0.45,
        ),
        OverlayPrimitive(
            key="face_label_leader",
            points=(label_anchor, Point(face.x + corner_length, face.y)),
            color=NEON_CYAN,
            thickness=1,
        ),
        OverlayPrimitive(
            key="face_baseline",
            points=(Point(face.x + baseline_margin, baseline_y), Point(face.right - baseline_margin, baseline_y)),
            color=NEON_BLUE,
            thickness=1,
        ),
    )


def classify_hand_gesture(fingers_up: Iterable[int]) -> str:
    fingers = tuple(int(value) for value in fingers_up)
    if len(fingers) != 5:
        raise ValueError("fingers_up must contain exactly five values")
    if any(value not in (0, 1) for value in fingers):
        raise ValueError("fingers_up values must be 0 or 1")

    named_gestures = {
        (1, 1, 1, 1, 1): "PALM",
        (0, 0, 0, 0, 0): "FIST",
        (0, 1, 0, 0, 0): "POINT",
        (0, 1, 1, 0, 0): "PEACE",
        (1, 0, 0, 0, 0): "THUMB",
    }
    return named_gestures.get(fingers, f"{sum(fingers)} FINGERS")


def build_hand_gesture_overlay(detection: DetectedHand) -> tuple[OverlayPrimitive, ...]:
    palm = detection.palm
    label_anchor = Point(palm.x, max(12, palm.y - 6))
    gesture = classify_hand_gesture(detection.fingers_up)
    label_primitives = (
        OverlayPrimitive(
            key="hand_gesture_label",
            points=(label_anchor,),
            color=NEON_MAGENTA,
            thickness=1,
            label=gesture,
            font_scale=0.45,
        ),
        OverlayPrimitive(
            key="hand_gesture_leader",
            points=(label_anchor, palm.center),
            color=NEON_MAGENTA,
            thickness=1,
        ),
        OverlayPrimitive(
            key="hand_center_dot",
            points=(palm.center,),
            color=NEON_CYAN,
            thickness=2,
            radius=3,
        ),
    )
    if len(detection.landmarks) < 21:
        return label_primitives

    landmarks = detection.landmarks
    return (
        OverlayPrimitive(
            key="hand_palm_outline",
            points=(landmarks[0], landmarks[1], landmarks[5], landmarks[9], landmarks[13], landmarks[17], landmarks[0]),
            color=NEON_MAGENTA,
            thickness=2,
        ),
        OverlayPrimitive(
            key="hand_thumb",
            points=(landmarks[1], landmarks[2], landmarks[3], landmarks[4]),
            color=NEON_CYAN,
            thickness=2,
        ),
        OverlayPrimitive(
            key="hand_index",
            points=(landmarks[5], landmarks[6], landmarks[7], landmarks[8]),
            color=NEON_CYAN,
            thickness=2,
        ),
        OverlayPrimitive(
            key="hand_middle",
            points=(landmarks[9], landmarks[10], landmarks[11], landmarks[12]),
            color=NEON_CYAN,
            thickness=2,
        ),
        OverlayPrimitive(
            key="hand_ring",
            points=(landmarks[13], landmarks[14], landmarks[15], landmarks[16]),
            color=NEON_CYAN,
            thickness=2,
        ),
        OverlayPrimitive(
            key="hand_pinky",
            points=(landmarks[17], landmarks[18], landmarks[19], landmarks[20]),
            color=NEON_CYAN,
            thickness=2,
        ),
        *label_primitives,
    )

__all__ = (
    "NEON_BLUE",
    "NEON_CYAN",
    "NEON_MAGENTA",
    "Point",
    "Rect",
    "DetectedFace",
    "DetectedHand",
    "OverlayPrimitive",
    "build_biometrics_overlay",
    "classify_hand_gesture",
    "build_hand_gesture_overlay",
)
