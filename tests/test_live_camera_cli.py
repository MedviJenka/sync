import contextlib
import io
import unittest
from unittest.mock import patch

import main
from vision.engine.live_camera import CVZoneUnavailableError, OpenCVUnavailableError


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


if __name__ == "__main__":
    unittest.main()
