"""Live OpenCV camera runner for the biometrics overlay."""

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


DEFAULT_WINDOW_TITLE = "Live Biometrics Overlay"
DEFAULT_OVERLAY_BGR = (255, 255, 0)


class OpenCVUnavailableError(RuntimeError):
    """Raised when OpenCV is required for live camera support but unavailable."""


class CVZoneUnavailableError(RuntimeError):
    """Raised when CVZone is required for stylized overlay rendering but unavailable."""


def require_cv2() -> Any:
    """Return the lazily imported OpenCV module or raise a clear install error."""

    try:
        import cv2  # type: ignore[import-not-found]
    except ImportError as exc:
        raise OpenCVUnavailableError(
            "Live camera support requires OpenCV. Install it with: "
            "pip install opencv-python"
        ) from exc

    return cv2


def require_cvzone() -> Any:
    """Return the lazily imported CVZone module or raise a clear install error."""

    try:
        import cvzone  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CVZoneUnavailableError(
            "Stylized camera overlays require CVZone. Install it with: "
            "pip install cvzone"
        ) from exc

    return cvzone


def require_hand_detector() -> Any:
    """Return the lazily imported CVZone hand detector or raise a clear install error."""

    try:
        from cvzone.HandTrackingModule import HandDetector  # type: ignore[import-not-found]
    except ImportError as exc:
        raise CVZoneUnavailableError(
            "Hand gesture support requires CVZone and MediaPipe. Install them with: "
            "pip install cvzone mediapipe"
        ) from exc

    return HandDetector


def run_live_camera(camera_index: int = 0, window_title: str = DEFAULT_WINDOW_TITLE) -> None:
    """Open a camera, detect faces, and render a low-clutter tracking overlay."""

    cv2 = require_cv2()
    cvzone = require_cvzone()
    overlay = _load_overlay_api()
    face_cascade = _load_cascade(cv2, "haarcascade_frontalface_default.xml")
    hand_detector = _load_hand_detector()

    capture = cv2.VideoCapture(camera_index)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open camera at index {camera_index}")

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("Could not read a frame from the camera")

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
            )

            for face in faces:
                detection = _to_detected_face(face, overlay)
                primitives = overlay.build_biometrics_overlay(detection)
                _draw_overlay(cv2, cvzone, frame, primitives)

            hands, frame = hand_detector.findHands(frame, draw=False, flipType=True)
            for hand in hands:
                detection = _to_detected_hand(hand, hand_detector.fingersUp(hand), overlay)
                primitives = overlay.build_hand_gesture_overlay(detection)
                _draw_overlay(cv2, cvzone, frame, primitives)

            cv2.imshow(window_title, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        capture.release()
        cv2.destroyWindow(window_title)


def _load_overlay_api() -> Any:
    from . import overlay

    return overlay


def _load_hand_detector() -> Any:
    hand_detector = require_hand_detector()
    return hand_detector(maxHands=2, detectionCon=0.6, minTrackCon=0.5)


def _load_cascade(cv2: Any, filename: str) -> Any:
    cascade_path = Path(cv2.data.haarcascades) / filename
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        raise RuntimeError(f"Could not load OpenCV Haar cascade: {cascade_path}")
    return cascade


def _to_detected_face(face: Sequence[int], overlay: Any) -> Any:
    x, y, width, height = (int(value) for value in face)
    return overlay.DetectedFace(face=overlay.Rect(x=x, y=y, width=width, height=height))


def _to_detected_hand(hand: Mapping[str, Any], fingers_up: Sequence[int], overlay: Any) -> Any:
    landmarks = tuple(_landmark_to_point(landmark, overlay) for landmark in hand.get("lmList", ()))
    palm = (
        _palm_rect_from_landmarks(landmarks, overlay)
        if len(landmarks) > 17
        else _bbox_to_rect(hand["bbox"], overlay)
    )
    return overlay.DetectedHand(
        palm=palm,
        fingers_up=tuple(int(value) for value in fingers_up),
        landmarks=landmarks,
        handedness=hand.get("type"),
    )


def _landmark_to_point(landmark: Sequence[int], overlay: Any) -> Any:
    return overlay.Point(x=int(landmark[0]), y=int(landmark[1]))


def _palm_rect_from_landmarks(landmarks: Sequence[Any], overlay: Any) -> Any:
    palm_points = tuple(landmarks[index] for index in (0, 1, 5, 9, 13, 17))
    left = min(point.x for point in palm_points)
    top = min(point.y for point in palm_points)
    right = max(point.x for point in palm_points)
    bottom = max(point.y for point in palm_points)
    return overlay.Rect(x=left, y=top, width=max(1, right - left), height=max(1, bottom - top))


def _bbox_to_rect(bbox: Sequence[int], overlay: Any) -> Any:
    x, y, width, height = (int(value) for value in bbox)
    return overlay.Rect(x=x, y=y, width=width, height=height)


def _draw_overlay(cv2: Any, cvzone: Any, frame: Any, primitives: Iterable[Any]) -> None:
    for primitive in primitives:
        _draw_primitive(cv2, cvzone, frame, primitive)


def _draw_primitive(cv2: Any, cvzone: Any, frame: Any, primitive: Any) -> None:
    color = tuple(int(channel) for channel in getattr(primitive, "color", DEFAULT_OVERLAY_BGR))
    thickness = int(getattr(primitive, "thickness", 2))
    points = tuple(_point_to_xy(point) for point in primitive.points)
    if not points:
        return

    label = getattr(primitive, "label", None)
    if label:
        _draw_label(
            cv2,
            frame,
            label,
            points[0],
            color,
            thickness,
            float(getattr(primitive, "font_scale", 0.7)),
        )
        return

    if primitive.key == "face_hud_box" and len(points) == 2:
        start, end = points
        cvzone.cornerRect(
            frame,
            (start[0], start[1], end[0] - start[0], end[1] - start[1]),
            l=int(getattr(primitive, "corner_length", 0) or 30),
            t=thickness,
            rt=0,
            colorR=color,
            colorC=tuple(int(channel) for channel in (getattr(primitive, "accent_color", None) or color)),
        )
        return

    if len(points) == 1:
        cv2.circle(frame, points[0], int(getattr(primitive, "radius", 4) or 4), color, thickness)
        return

    for start, end in zip(points, points[1:]):
        cv2.line(frame, start, end, color, thickness)


def _draw_label(
    cv2: Any,
    frame: Any,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    font_scale: float,
) -> None:
    font = getattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
    shadow_origin = (origin[0] + 1, origin[1] + 1)
    shadow_thickness = max(thickness + 1, 2)
    cv2.putText(frame, str(text), shadow_origin, font, font_scale, (0, 0, 0), shadow_thickness)
    cv2.putText(frame, str(text), origin, font, font_scale, color, thickness)


def _point_to_xy(point: Any) -> tuple[int, int]:
    return (int(point.x), int(point.y))


__all__ = (
    "CVZoneUnavailableError",
    "OpenCVUnavailableError",
    "require_cv2",
    "require_cvzone",
    "require_hand_detector",
    "run_live_camera",
)
