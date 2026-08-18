import contextlib
import io
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import drone
import drone.cli as main
from drone.vision.engine.live_camera import CVZoneUnavailableError, OpenCVUnavailableError, PySimVerseUnavailableError


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
            required_stable_detections=2,
            command_cooldown_detections=3,
            lost_tracking_land_detections=30,
            start_armed=True,
        )

    def test_forwards_safety_configuration_from_cli(self):
        with patch.object(main, "run_live_camera") as run_live_camera:
            main.main(["--stable-detections", "4", "--command-cooldown-detections", "9"])

        run_live_camera.assert_called_once_with(
            camera_index=0,
            window_title=main.DEFAULT_WINDOW_TITLE,
            enable_drone_control=True,
            required_stable_detections=4,
            command_cooldown_detections=9,
            lost_tracking_land_detections=30,
            start_armed=True,
        )

    def test_can_start_disarmed_from_cli(self):
        with patch.object(main, "run_live_camera") as run_live_camera:
            main.main(["--start-disarmed"])

        run_live_camera.assert_called_once_with(
            camera_index=0,
            window_title=main.DEFAULT_WINDOW_TITLE,
            enable_drone_control=True,
            required_stable_detections=2,
            command_cooldown_detections=3,
            lost_tracking_land_detections=30,
            start_armed=False,
        )

    def test_forwards_phase_two_safety_configuration_from_cli(self):
        with patch.object(main, "run_live_camera") as run_live_camera:
            main.main(["--start-armed", "--lost-tracking-land-detections", "12"])

        run_live_camera.assert_called_once_with(
            camera_index=0,
            window_title=main.DEFAULT_WINDOW_TITLE,
            enable_drone_control=True,
            required_stable_detections=2,
            command_cooldown_detections=3,
            lost_tracking_land_detections=12,
            start_armed=True,
        )

    def test_rejects_invalid_safety_configuration(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as exit_context:
                main.main(["--stable-detections", "0"])

        self.assertEqual(exit_context.exception.code, 2)
        self.assertIn("--stable-detections must be at least 1", stderr.getvalue())

    def test_console_entrypoint_delegates_to_live_camera_cli(self):
        with patch.object(main, "main") as camera_main:
            drone.main(["--no-drone-control"])

        camera_main.assert_called_once_with(["--no-drone-control"])

    def test_drone_loader_passes_safety_configuration_to_controller(self):
        from drone.vision.engine import live_camera

        controller_class = MagicMock()
        fake_pysimverse = types.SimpleNamespace(Drone=FakePySimVerseDrone)
        with patch.dict(sys.modules, {"pysimverse": fake_pysimverse}):
            with patch.object(live_camera, "DroneGestureController", controller_class, create=True):
                live_camera._load_drone_gesture_controller(
                    required_stable_detections=5,
                    command_cooldown_detections=7,
                    lost_tracking_land_detections=11,
                    start_armed=True,
                )

        controller_class.assert_called_once()
        _, kwargs = controller_class.call_args
        self.assertEqual(kwargs["required_stable_detections"], 5)
        self.assertEqual(kwargs["command_cooldown_detections"], 7)
        self.assertEqual(kwargs["lost_tracking_land_detections"], 11)
        self.assertTrue(kwargs["start_armed"])

    def test_keyboard_emergency_land_invokes_controller(self):
        from drone.vision.engine import live_camera

        controller = FakeDroneGestureController()
        with _patched_live_camera(live_camera, controller, wait_keys=(ord("e"), ord("q"))):
            live_camera.run_live_camera()

        self.assertEqual(controller.emergency_land_calls, 1)

    def test_camera_read_failure_requests_emergency_land_before_raising(self):
        from drone.vision.engine import live_camera

        controller = FakeDroneGestureController()
        with _patched_live_camera(live_camera, controller, frames=()):
            with self.assertRaises(RuntimeError):
                live_camera.run_live_camera()

        self.assertEqual(controller.emergency_land_calls, 1)

    def test_missing_hands_report_tracking_loss_to_controller(self):
        from drone.vision.engine import live_camera

        controller = FakeDroneGestureController()
        with _patched_live_camera(live_camera, controller, wait_keys=(0, ord("q"))):
            live_camera.run_live_camera()

        self.assertEqual(controller.tracking_lost_calls, 2)

    def test_live_camera_draws_drone_status_feedback(self):
        from drone.vision.engine import live_camera

        controller = FakeDroneGestureController(status_text="DISARMED - PINCH TO ARM")
        draw_overlay = MagicMock()
        with _patched_live_camera(live_camera, controller, wait_keys=(ord("q"),)):
            with patch.object(live_camera, "_draw_overlay", draw_overlay):
                live_camera.run_live_camera()

        drawn_keys = [primitive.key for call in draw_overlay.call_args_list for primitive in call.args[3]]
        self.assertIn("drone_status_label", drawn_keys)

    def test_drone_controller_failure_requests_emergency_land_before_raising(self):
        from drone.vision.engine import live_camera

        controller = FakeDroneGestureController(detection_error=RuntimeError("drone disconnected"))
        with _patched_live_camera(live_camera, controller, hands=(_fake_hand(),)):
            with self.assertRaises(RuntimeError):
                live_camera.run_live_camera()

        self.assertEqual(controller.emergency_land_calls, 1)



@contextlib.contextmanager
def _patched_live_camera(
    live_camera,
    controller,
    *,
    wait_keys=(ord("q"),),
    frames=(object(), object()),
    hands=(),
):
    fake_cv2 = FakeLiveCV2(wait_keys=wait_keys, frames=frames)
    hand_detector = FakeHandDetector(hands=hands)
    with patch.object(live_camera, "require_cv2", return_value=fake_cv2), patch.object(
        live_camera, "require_cvzone", return_value=types.SimpleNamespace()
    ), patch.object(live_camera, "_load_cascade", return_value=FakeCascade()), patch.object(
        live_camera, "_load_hand_detector", return_value=hand_detector
    ), patch.object(
        live_camera, "_load_drone_gesture_controller", return_value=controller
    ):
        yield


class FakeLiveCV2:
    COLOR_BGR2GRAY = 0

    def __init__(self, *, wait_keys, frames):
        self.data = types.SimpleNamespace(haarcascades="")
        self._wait_keys = list(wait_keys)
        self._capture = FakeCapture(frames)

    def VideoCapture(self, camera_index):
        return self._capture

    def cvtColor(self, frame, color):
        return frame

    def imshow(self, window_title, frame):
        pass

    def waitKey(self, delay):
        return self._wait_keys.pop(0) if self._wait_keys else ord("q")

    def destroyWindow(self, window_title):
        pass

    def putText(self, frame, text, origin, font, scale, color, thickness):
        pass


class FakeCapture:
    def __init__(self, frames):
        self._frames = list(frames)

    def isOpened(self):
        return True

    def read(self):
        if not self._frames:
            return False, None
        return True, self._frames.pop(0)

    def release(self):
        pass


class FakeCascade:
    def detectMultiScale(self, *args, **kwargs):
        return ()


class FakeHandDetector:
    def __init__(self, *, hands=()):
        self._hands = tuple(hands)

    def findHands(self, frame, draw=False, flipType=True):
        return self._hands, frame

    def fingersUp(self, hand):
        return hand.get("fingers_up", (0, 1, 0, 0, 0))


class FakeDroneGestureController:
    def __init__(self, detection_error=None, status_text="ARMED"):
        self.emergency_land_calls = 0
        self.tracking_lost_calls = 0
        self.status_text = status_text
        self._detection_error = detection_error

    def handle_detection(self, detection):
        if self._detection_error is not None:
            raise self._detection_error

    def handle_tracking_lost(self):
        self.tracking_lost_calls += 1

    def emergency_land(self):
        self.emergency_land_calls += 1



def _fake_hand():
    landmarks = [[60, 120, 0] for _ in range(21)]
    landmarks[5] = [60, 95, 0]
    landmarks[8] = [60, 20, 0]
    return {"bbox": (42, 70, 60, 60), "lmList": landmarks, "fingers_up": (0, 1, 0, 0, 0)}

class FakePySimVerseDrone:
    def connect(self):
        pass

if __name__ == "__main__":
    unittest.main()
