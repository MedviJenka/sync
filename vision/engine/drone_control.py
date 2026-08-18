"""Translate camera hand gestures into PySimVerse drone commands."""

from __future__ import annotations

from threading import Lock, Thread
from typing import Literal, Protocol

from .overlay import DetectedHand, Point

DroneCommand = Literal["takeoff", "land"]

INDEX_DIRECTION_MIN_PALM_RATIO = 0.55
INDEX_EXTENSION_MIN_PALM_RATIO = 0.75
ABSOLUTE_DIRECTION_MIN_PIXELS = 18.0


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
    """Runs takeoff/land commands once per valid pointing gesture state."""

    def __init__(
        self,
        drone: ControllableDrone,
        *,
        async_commands: bool = True,
        takeoff_height: int = 100,
        takeoff_speed: int = 25,
        landing_speed: int = 25,
    ) -> None:
        self._drone = drone
        self._async_commands = async_commands
        self._takeoff_height = takeoff_height
        self._takeoff_speed = takeoff_speed
        self._landing_speed = landing_speed
        self._lock = Lock()
        self._command_in_flight = False

    def handle_detection(self, detection: DetectedHand) -> DroneCommand | None:
        command = drone_command_from_hand(detection)
        if command is None or not self._should_execute(command):
            return None

        if self._async_commands:
            with self._lock:
                if self._command_in_flight:
                    return None
                self._command_in_flight = True
            Thread(target=self._execute_command, args=(command,), daemon=True).start()
        else:
            self._execute_command(command)
        return command

    def _should_execute(self, command: DroneCommand) -> bool:
        is_flying = bool(getattr(self._drone, "is_flying", False))
        return (command == "takeoff" and not is_flying) or (command == "land" and is_flying)

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


def _palm_scale(landmarks: tuple[Point, ...]) -> float:
    return max(_distance(landmarks[5], landmarks[17]), _distance(landmarks[0], landmarks[9]), 1.0)


def _distance(first: Point, second: Point) -> float:
    return ((first.x - second.x) ** 2 + (first.y - second.y) ** 2) ** 0.5


__all__ = (
    "ControllableDrone",
    "DroneCommand",
    "DroneGestureController",
    "drone_command_from_hand",
)
