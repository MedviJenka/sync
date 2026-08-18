import unittest

from vision.engine import DetectedHand, Point, Rect
from vision.engine.drone_control import DroneGestureController, drone_command_from_hand


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

    def test_controller_takes_off_once_when_grounded(self):
        drone = FakeDrone(is_flying=False)
        controller = DroneGestureController(drone, async_commands=False)

        command = controller.handle_detection(
            _pointing_hand(index_tip=Point(x=60, y=20), index_mcp=Point(x=60, y=95), fingers_up=(0, 1, 0, 0, 0))
        )
        repeated = controller.handle_detection(
            _pointing_hand(index_tip=Point(x=60, y=20), index_mcp=Point(x=60, y=95), fingers_up=(0, 1, 0, 0, 0))
        )

        self.assertEqual(command, "takeoff")
        self.assertIsNone(repeated)
        self.assertEqual(drone.takeoff_calls, [(100, 25)])
        self.assertEqual(drone.land_calls, [])

    def test_controller_lands_once_when_flying(self):
        drone = FakeDrone(is_flying=True)
        controller = DroneGestureController(drone, async_commands=False)

        command = controller.handle_detection(
            _pointing_hand(index_tip=Point(x=60, y=170), index_mcp=Point(x=60, y=95), fingers_up=(0, 0, 0, 0, 0))
        )
        repeated = controller.handle_detection(
            _pointing_hand(index_tip=Point(x=60, y=170), index_mcp=Point(x=60, y=95), fingers_up=(0, 0, 0, 0, 0))
        )

        self.assertEqual(command, "land")
        self.assertIsNone(repeated)
        self.assertEqual(drone.takeoff_calls, [])
        self.assertEqual(drone.land_calls, [25])


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
