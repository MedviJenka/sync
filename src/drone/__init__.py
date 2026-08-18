from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from main import main as camera_main

    camera_main(argv)
