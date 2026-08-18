import argparse
import sys
from collections.abc import Sequence

from vision.engine.live_camera import (
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
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        run_live_camera(
            camera_index=args.camera_index,
            window_title=args.window_title,
            enable_drone_control=args.drone_control,
        )
    except (CVZoneUnavailableError, OpenCVUnavailableError, PySimVerseUnavailableError) as error:
        print(error, file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == "__main__":
    main()
