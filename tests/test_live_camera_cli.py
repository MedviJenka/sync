import contextlib
import io
import unittest
from unittest.mock import patch

import drone
import main
from vision.engine.live_camera import CVZoneUnavailableError, OpenCVUnavailableError, PySimVerseUnavailableError


class LiveCameraCliTest(unittest.TestCase):
    def test_prints_dependency_error_without_traceback(self):
        stderr = io.StringIO()
        with patch.object(main, "run_live_camera", side_effect=OpenCVUnavailableError("install opencv-python")):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exit_context:
                    main.main([])

        self.assertEqual(exit_context.exception.code, 1)
        self.assertIn("install opencv-python", stderr.getvalue())

    def test_prints_overlay_dependency_error_without_traceback(self):
        stderr = io.StringIO()
        with patch.object(main, "run_live_camera", side_effect=CVZoneUnavailableError("install cvzone mediapipe")):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exit_context:
                    main.main([])

        self.assertEqual(exit_context.exception.code, 1)
        self.assertIn("install cvzone mediapipe", stderr.getvalue())

    def test_prints_pysimverse_dependency_error_without_traceback(self):
        stderr = io.StringIO()
        with patch.object(main, "run_live_camera", side_effect=PySimVerseUnavailableError("install pysimverse")):
            with contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as exit_context:
                    main.main([])

        self.assertEqual(exit_context.exception.code, 1)
        self.assertIn("install pysimverse", stderr.getvalue())

    def test_can_disable_drone_control_from_cli(self):
        with patch.object(main, "run_live_camera") as run_live_camera:
            main.main(["--no-drone-control"])

        run_live_camera.assert_called_once_with(
            camera_index=0,
            window_title=main.DEFAULT_WINDOW_TITLE,
            enable_drone_control=False,
        )

    def test_console_entrypoint_delegates_to_live_camera_cli(self):
        with patch.object(main, "main") as camera_main:
            drone.main(["--no-drone-control"])

        camera_main.assert_called_once_with(["--no-drone-control"])


if __name__ == "__main__":
    unittest.main()
