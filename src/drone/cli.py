import argparse
import sys
from collections.abc import Sequence
from drone.vision.engine.drone_control import DEFAULT_COMMAND_COOLDOWN_DETECTIONS, DEFAULT_LOST_TRACKING_LAND_DETECTIONS, DEFAULT_REQUIRED_STABLE_DETECTIONS
from drone.vision.engine.live_camera import (
    CVZoneUnavailableError,
    DEFAULT_WINDOW_TITLE,
    OpenCVUnavailableError,
    PySimVerseUnavailableError,
    run_live_camera,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the live biometrics camera overlay.")
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Camera index passed to OpenCV VideoCapture (default: 0).",
    )
    parser.add_argument(
        "--window-title",
        default=DEFAULT_WINDOW_TITLE,
        help=f"Camera window title (default: {DEFAULT_WINDOW_TITLE!r}).",
    )
    parser.add_argument(
        "--no-drone-control",
        action="store_false",
        dest="drone_control",
        help="Disable PySimVerse takeoff/land control from pointing hand gestures.",
    )
    parser.add_argument(
        "--stable-detections",
        type=_at_least_one,
        default=DEFAULT_REQUIRED_STABLE_DETECTIONS,
        help=(
            "Consecutive matching hand detections required before takeoff/land "
            f"(default: {DEFAULT_REQUIRED_STABLE_DETECTIONS})."
        ),
    )
    parser.add_argument(
        "--command-cooldown-detections",
        type=_non_negative_int,
        default=DEFAULT_COMMAND_COOLDOWN_DETECTIONS,
        help=(
            "Detections to ignore after a drone command before accepting another "
            f"(default: {DEFAULT_COMMAND_COOLDOWN_DETECTIONS})."
        ),
    )
    parser.add_argument(
        "--lost-tracking-land-detections",
        type=_at_least_one,
        default=DEFAULT_LOST_TRACKING_LAND_DETECTIONS,
        help=(
            "Consecutive no-hand detections before emergency landing "
            f"(default: {DEFAULT_LOST_TRACKING_LAND_DETECTIONS})."
        ),
    )
    flight_mode = parser.add_mutually_exclusive_group()
    flight_mode.add_argument(
        "--start-armed",
        action="store_true",
        dest="start_armed",
        default=True,
        help="Start with gesture flight commands armed. This is the default.",
    )
    flight_mode.add_argument(
        "--start-disarmed",
        action="store_false",
        dest="start_armed",
        help="Require a pinch gesture before takeoff/land commands are accepted.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        run_live_camera(
            camera_index=args.camera_index,
            window_title=args.window_title,
            enable_drone_control=args.drone_control,
            required_stable_detections=args.stable_detections,
            command_cooldown_detections=args.command_cooldown_detections,
            lost_tracking_land_detections=args.lost_tracking_land_detections,
            start_armed=args.start_armed,
        )
    except (CVZoneUnavailableError, OpenCVUnavailableError, PySimVerseUnavailableError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error

def _at_least_one(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("--stable-detections must be at least 1")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("--command-cooldown-detections must be non-negative")
    return parsed


if __name__ == "__main__":
    main()
