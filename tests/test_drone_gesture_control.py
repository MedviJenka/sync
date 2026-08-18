import unittest

from drone.vision.engine import DetectedHand, Point, Rect
from drone.vision.engine.drone_control import DroneGestureController, drone_command_from_hand


class DroneGestureControlTest(unittest.TestCase):
    def test_pointing_up_index_finger_requests_takeoff(self):
        detection = _pointing_hand(index_tip=Point(x=60, y=20), index_mcp=Point(x=60, y=95), fingers_up=(0, 1, 0, 0, 0))

        self.assertEqual(drone_command_from_hand(detection), "takeoff")

    def test_pointing_down_index_finger_requests_landing_even_when_cvzone_marks_index_down(self):
        detection = _pointing_hand(index_tip=Point(x=60, y=170), index_mcp=Point(x=60, y=95), fingers_up=(0, 0, 0, 0, 0))

        self.assertEqual(drone_command_from_hand(detection), "land")

    def test_sideways_or_open_hand_does_not_request_drone_command(self):
        sideways = _pointing_hand(index_tip=Point(x=135, y=95), index_mcp=Point(x=60, y=95), fingers_up=(0, 1, 0, 0, 0))
        open_hand = _pointing_hand(index_tip=Point(x=60, y=20), index_mcp=Point(x=60, y=95), fingers_up=(1, 1, 1, 1, 1))

        self.assertIsNone(drone_command_from_hand(sideways))
        self.assertIsNone(drone_command_from_hand(open_hand))

    def test_controller_ignores_flight_command_until_armed(self):
        drone = FakeDrone(is_flying=False)
        controller = DroneGestureController(drone, async_commands=False, required_stable_detections=1)
        detection = _pointing_hand(index_tip=Point(x=60, y=20), index_mcp=Point(x=60, y=95), fingers_up=(0, 1, 0, 0, 0))

        self.assertIsNone(controller.handle_detection(detection))
        self.assertEqual(drone.takeoff_calls, [])

    def test_default_status_explains_why_takeoff_is_ignored(self):
        drone = FakeDrone(is_flying=False)
        controller = DroneGestureController(drone, async_commands=False, required_stable_detections=1)

        self.assertEqual(controller.status_text, "DISARMED - PINCH TO ARM")

    def test_armed_status_explains_takeoff_and_land_gestures(self):
        drone = FakeDrone(is_flying=False)
        controller = DroneGestureController(drone, async_commands=False, required_stable_detections=1, start_armed=True)

        self.assertEqual(controller.status_text, "ARMED - POINT UP TAKEOFF / POINT DOWN LAND")

    def test_pinch_toggles_armed_state_without_moving_drone(self):
        drone = FakeDrone(is_flying=False)
        controller = DroneGestureController(drone, async_commands=False, required_stable_detections=1)

        self.assertFalse(controller.is_armed)
        self.assertEqual(controller.handle_detection(_pinch_hand()), "arm")
        self.assertTrue(controller.is_armed)
        self.assertEqual(controller.handle_detection(_pinch_hand()), "disarm")
        self.assertFalse(controller.is_armed)
        self.assertEqual(drone.takeoff_calls, [])
        self.assertEqual(drone.land_calls, [])

    def test_controller_requires_stable_takeoff_gesture_before_commanding_drone(self):
        drone = FakeDrone(is_flying=False)
        controller = DroneGestureController(drone, async_commands=False, required_stable_detections=2, start_armed=True)
        detection = _pointing_hand(index_tip=Point(x=60, y=20), index_mcp=Point(x=60, y=95), fingers_up=(0, 1, 0, 0, 0))

        first_frame = controller.handle_detection(detection)
        second_frame = controller.handle_detection(detection)
        repeated = controller.handle_detection(detection)

        self.assertIsNone(first_frame)
        self.assertEqual(second_frame, "takeoff")
        self.assertIsNone(repeated)
        self.assertEqual(drone.takeoff_calls, [(100, 25)])
        self.assertEqual(drone.land_calls, [])

    def test_controller_requires_stable_landing_gesture_before_commanding_drone(self):
        drone = FakeDrone(is_flying=True)
        controller = DroneGestureController(drone, async_commands=False, required_stable_detections=2, start_armed=True)
        detection = _pointing_hand(index_tip=Point(x=60, y=170), index_mcp=Point(x=60, y=95), fingers_up=(0, 0, 0, 0, 0))

        first_frame = controller.handle_detection(detection)
        second_frame = controller.handle_detection(detection)
        repeated = controller.handle_detection(detection)

        self.assertIsNone(first_frame)
        self.assertEqual(second_frame, "land")
        self.assertIsNone(repeated)
        self.assertEqual(drone.takeoff_calls, [])
        self.assertEqual(drone.land_calls, [25])
 
    def test_unstable_gesture_sequence_resets_command_stability(self):
        drone = FakeDrone(is_flying=False)
        controller = DroneGestureController(drone, async_commands=False, required_stable_detections=2, start_armed=True)
        takeoff = _pointing_hand(index_tip=Point(x=60, y=20), index_mcp=Point(x=60, y=95), fingers_up=(0, 1, 0, 0, 0))
        sideways = _pointing_hand(index_tip=Point(x=135, y=95), index_mcp=Point(x=60, y=95), fingers_up=(0, 1, 0, 0, 0))

        self.assertIsNone(controller.handle_detection(takeoff))
        self.assertIsNone(controller.handle_detection(sideways))
        self.assertIsNone(controller.handle_detection(takeoff))
        self.assertEqual(controller.handle_detection(takeoff), "takeoff")
        self.assertEqual(drone.takeoff_calls, [(100, 25)])

    def test_command_cooldown_suppresses_opposite_command_after_takeoff(self):
        drone = FakeDrone(is_flying=False)
        controller = DroneGestureController(
            drone,
            async_commands=False,
            required_stable_detections=1,
            command_cooldown_detections=2,
            start_armed=True,
        )
        takeoff = _pointing_hand(index_tip=Point(x=60, y=20), index_mcp=Point(x=60, y=95), fingers_up=(0, 1, 0, 0, 0))
        land = _pointing_hand(index_tip=Point(x=60, y=170), index_mcp=Point(x=60, y=95), fingers_up=(0, 0, 0, 0, 0))

        self.assertEqual(controller.handle_detection(takeoff), "takeoff")
        self.assertIsNone(controller.handle_detection(land))
        self.assertIsNone(controller.handle_detection(land))
        self.assertEqual(controller.handle_detection(land), "land")
        self.assertEqual(drone.land_calls, [25])

    def test_open_palm_emergency_lands_even_when_disarmed_and_cooling_down(self):
        drone = FakeDrone(is_flying=False)
        controller = DroneGestureController(
            drone,
            async_commands=False,
            required_stable_detections=1,
            command_cooldown_detections=5,
            start_armed=True,
        )
        takeoff = _pointing_hand(index_tip=Point(x=60, y=20), index_mcp=Point(x=60, y=95), fingers_up=(0, 1, 0, 0, 0))

        self.assertEqual(controller.handle_detection(takeoff), "takeoff")
        controller.disarm()
        self.assertEqual(controller.handle_detection(_open_palm_hand()), "emergency_land")
        self.assertFalse(controller.is_armed)
        self.assertEqual(drone.land_calls, [25])

    def test_tracking_loss_emergency_lands_after_threshold(self):
        drone = FakeDrone(is_flying=True)
        controller = DroneGestureController(
            drone,
            async_commands=False,
            lost_tracking_land_detections=2,
            start_armed=False,
        )

        self.assertIsNone(controller.handle_tracking_lost())
        self.assertEqual(controller.handle_tracking_lost(), "emergency_land")
        self.assertEqual(drone.land_calls, [25])



def _pinch_hand() -> DetectedHand:
    landmarks = [Point(x=60, y=120) for _ in range(21)]
    landmarks[0] = Point(x=60, y=130)
    landmarks[4] = Point(x=72, y=62)
    landmarks[5] = Point(x=64, y=108)
    landmarks[8] = Point(x=75, y=64)
    landmarks[9] = Point(x=80, y=80)
    landmarks[17] = Point(x=96, y=114)
    return DetectedHand(
        palm=Rect(x=42, y=70, width=60, height=60),
        fingers_up=(1, 1, 0, 0, 0),
        landmarks=tuple(landmarks),
    )


def _open_palm_hand() -> DetectedHand:
    return _pointing_hand(
        index_tip=Point(x=60, y=20),
        index_mcp=Point(x=60, y=95),
        fingers_up=(1, 1, 1, 1, 1),
    )

def _pointing_hand(*, index_tip: Point, index_mcp: Point, fingers_up: tuple[int, int, int, int, int]) -> DetectedHand:
    landmarks = [Point(x=60, y=120) for _ in range(21)]
    landmarks[0] = Point(x=60, y=130)
    landmarks[1] = Point(x=54, y=114)
    landmarks[4] = Point(x=48, y=120)
    landmarks[5] = index_mcp
    landmarks[6] = Point(x=(index_mcp.x + index_tip.x) // 2, y=(index_mcp.y + index_tip.y) // 2)
    landmarks[7] = Point(x=(index_mcp.x + index_tip.x) // 2, y=(index_mcp.y + index_tip.y) // 2)
    landmarks[8] = index_tip
    landmarks[9] = Point(x=70, y=102)
    landmarks[12] = Point(x=72, y=115)
    landmarks[13] = Point(x=84, y=108)
    landmarks[16] = Point(x=84, y=119)
    landmarks[17] = Point(x=96, y=114)
    landmarks[20] = Point(x=94, y=123)
    return DetectedHand(
        palm=Rect(x=42, y=95, width=60, height=45),
        fingers_up=fingers_up,
        landmarks=tuple(landmarks),
    )


class FakeDrone:
    def __init__(self, *, is_flying: bool) -> None:
        self.is_flying = is_flying
        self.takeoff_calls = []
        self.land_calls = []

    def take_off(self, takeoff_height=100, takeoff_speed=25):
        self.takeoff_calls.append((takeoff_height, takeoff_speed))
        self.is_flying = True

    def land(self, landing_speed=25):
        self.land_calls.append(landing_speed)
        self.is_flying = False


if __name__ == "__main__":
    unittest.main()
