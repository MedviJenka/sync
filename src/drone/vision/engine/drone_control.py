"""Translate camera hand gestures into PySimVerse drone commands."""

from __future__ import annotations

from threading import Lock, Thread
from typing import Literal, Protocol

from .overlay import DetectedHand, Point, classify_hand_gesture

DroneCommand = Literal["takeoff", "land"]
ControllerAction = Literal["arm", "disarm", "takeoff", "land", "emergency_land"]

INDEX_DIRECTION_MIN_PALM_RATIO = 0.55
INDEX_EXTENSION_MIN_PALM_RATIO = 0.75
ABSOLUTE_DIRECTION_MIN_PIXELS = 18.0
DEFAULT_REQUIRED_STABLE_DETECTIONS = 2
DEFAULT_COMMAND_COOLDOWN_DETECTIONS = 3
DEFAULT_LOST_TRACKING_LAND_DETECTIONS = 30



class ControllableDrone(Protocol):
    is_flying: bool

    def take_off(self, takeoff_height=100, takeoff_speed=25): ...

    def land(self, landing_speed=25): ...


def drone_command_from_hand(detection: DetectedHand) -> DroneCommand | None:
    """Return the drone command represented by an index-finger up/down gesture."""

    if len(detection.landmarks) < 21 or not _is_index_only_point(detection):
        return None

    landmarks = detection.landmarks
    index_mcp = landmarks[5]
    index_tip = landmarks[8]
    palm_scale = _palm_scale(landmarks)
    vertical_delta = index_tip.y - index_mcp.y
    horizontal_delta = abs(index_tip.x - index_mcp.x)
    threshold = max(palm_scale * INDEX_DIRECTION_MIN_PALM_RATIO, ABSOLUTE_DIRECTION_MIN_PIXELS)

    if abs(vertical_delta) <= horizontal_delta or abs(vertical_delta) < threshold:
        return None
    if vertical_delta < 0:
        return "takeoff"
    return "land"


class DroneGestureController:
    """Runs armed takeoff/land commands and emergency land safety actions."""

    def __init__(
        self,
        drone: ControllableDrone,
        *,
        async_commands: bool = True,
        takeoff_height: int = 100,
        takeoff_speed: int = 25,
        landing_speed: int = 25,
        required_stable_detections: int = DEFAULT_REQUIRED_STABLE_DETECTIONS,
        command_cooldown_detections: int = DEFAULT_COMMAND_COOLDOWN_DETECTIONS,
        lost_tracking_land_detections: int = DEFAULT_LOST_TRACKING_LAND_DETECTIONS,
        start_armed: bool = False,
    ) -> None:
        if required_stable_detections < 1:
            raise ValueError("required_stable_detections must be at least 1")
        if command_cooldown_detections < 0:
            raise ValueError("command_cooldown_detections must be non-negative")
        if lost_tracking_land_detections < 1:
            raise ValueError("lost_tracking_land_detections must be at least 1")

        self._drone = drone
        self._async_commands = async_commands
        self._takeoff_height = takeoff_height
        self._takeoff_speed = takeoff_speed
        self._landing_speed = landing_speed
        self._required_stable_detections = required_stable_detections
        self._command_cooldown_detections = command_cooldown_detections
        self._lost_tracking_land_detections = lost_tracking_land_detections
        self._cooldown_remaining = 0
        self._pending_action: ControllerAction | None = None
        self._stable_detection_count = 0
        self._lost_tracking_count = 0
        self._is_armed = start_armed
        self._lock = Lock()
        self._command_in_flight = False

    @property
    def is_armed(self) -> bool:
        return self._is_armed

    @property
    def status_text(self) -> str:
        if not self._is_armed:
            return "DISARMED - PINCH TO ARM"
        return "ARMED - POINT UP TAKEOFF / POINT DOWN LAND"

    def arm(self) -> ControllerAction:
        self._is_armed = True
        self._reset_command_state()
        return "arm"

    def disarm(self) -> ControllerAction:
        self._is_armed = False
        self._reset_command_state()
        return "disarm"

    def handle_detection(self, detection: DetectedHand) -> ControllerAction | None:
        self._lost_tracking_count = 0

        if _is_emergency_land_gesture(detection):
            return self.emergency_land()
        if _is_arm_toggle_gesture(detection):
            if not self._is_stable_action("arm"):
                return None
            return self.disarm() if self._is_armed else self.arm()

        command = drone_command_from_hand(detection)
        if command is None:
            self._reset_stability()
            return None
        if not self._is_armed:
            self._reset_stability()
            return None
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            self._reset_stability()
            return None
        if not self._is_stable_action(command) or not self._should_execute(command):
            return None

        if not self._dispatch_drone_command(command):
            return None
        self._record_executed_command()
        return command

    def handle_tracking_lost(self) -> ControllerAction | None:
        self._reset_stability()
        self._lost_tracking_count += 1
        if self._lost_tracking_count < self._lost_tracking_land_detections:
            return None
        return self.emergency_land()

    def emergency_land(self) -> ControllerAction | None:
        self._is_armed = False
        self._cooldown_remaining = 0
        self._reset_stability()
        if not bool(getattr(self._drone, "is_flying", False)):
            return None
        if not self._dispatch_drone_command("land"):
            return None
        return "emergency_land"

    def _should_execute(self, command: DroneCommand) -> bool:
        is_flying = bool(getattr(self._drone, "is_flying", False))
        return (command == "takeoff" and not is_flying) or (command == "land" and is_flying)

    def _is_stable_action(self, action: ControllerAction) -> bool:
        if action == self._pending_action:
            self._stable_detection_count += 1
        else:
            self._pending_action = action
            self._stable_detection_count = 1
        return self._stable_detection_count >= self._required_stable_detections

    def _record_executed_command(self) -> None:
        self._cooldown_remaining = self._command_cooldown_detections
        self._reset_stability()

    def _reset_command_state(self) -> None:
        self._cooldown_remaining = 0
        self._reset_stability()

    def _reset_stability(self) -> None:
        self._pending_action = None
        self._stable_detection_count = 0

    def _dispatch_drone_command(self, command: DroneCommand) -> bool:
        if self._async_commands:
            with self._lock:
                if self._command_in_flight:
                    return False
                self._command_in_flight = True
            Thread(target=self._execute_command, args=(command,), daemon=True).start()
        else:
            self._execute_command(command)
        return True

    def _execute_command(self, command: DroneCommand) -> None:
        try:
            if command == "takeoff":
                self._drone.take_off(self._takeoff_height, self._takeoff_speed)
            else:
                self._drone.land(self._landing_speed)
        finally:
            if self._async_commands:
                with self._lock:
                    self._command_in_flight = False


def _is_index_only_point(detection: DetectedHand) -> bool:
    fingers = tuple(int(value) for value in detection.fingers_up)
    if len(fingers) != 5 or any(value not in (0, 1) for value in fingers):
        return False
    if any(fingers[index] for index in (0, 2, 3, 4)):
        return False
    if fingers[1] == 1:
        return True

    landmarks = detection.landmarks
    palm_scale = _palm_scale(landmarks)
    return _distance(landmarks[5], landmarks[8]) >= palm_scale * INDEX_EXTENSION_MIN_PALM_RATIO

def _is_arm_toggle_gesture(detection: DetectedHand) -> bool:
    return classify_hand_gesture(detection.fingers_up, detection.landmarks) == "PINCH"


def _is_emergency_land_gesture(detection: DetectedHand) -> bool:
    return tuple(detection.fingers_up) == (1, 1, 1, 1, 1)


def _palm_scale(landmarks: tuple[Point, ...]) -> float:
    return max(_distance(landmarks[5], landmarks[17]), _distance(landmarks[0], landmarks[9]), 1.0)


def _distance(first: Point, second: Point) -> float:
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5

__all__ = (
    "ABSOLUTE_DIRECTION_MIN_PIXELS",
    "ControllableDrone",
    "ControllerAction",
    "DEFAULT_COMMAND_COOLDOWN_DETECTIONS",
    "DEFAULT_LOST_TRACKING_LAND_DETECTIONS",
    "DEFAULT_REQUIRED_STABLE_DETECTIONS",
    "DroneCommand",
    "DroneGestureController",
    "drone_command_from_hand",
)
