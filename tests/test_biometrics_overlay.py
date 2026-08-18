import unittest

from vision.engine import (
    DetectedFace,
    DetectedHand,
    NEON_BLUE,
    NEON_CYAN,
    NEON_MAGENTA,
    OverlayPrimitive,
    Point,
    Rect,
    build_biometrics_overlay,
    build_hand_gesture_overlay,
    classify_hand_gesture,
)
from vision.engine.live_camera import (
    CVZoneUnavailableError,
    OpenCVUnavailableError,
    _draw_overlay,
    _to_detected_hand,
    require_cv2,
    require_cvzone,
)


class BiometricsOverlayTest(unittest.TestCase):
    def test_builds_clean_low_clutter_face_tracking_primitives(self):
        detection = DetectedFace(
            face=Rect(x=10, y=20, width=100, height=120),
            eyes=(
                Rect(x=30, y=55, width=18, height=12),
                Rect(x=72, y=56, width=18, height=12),
            ),
        )

        primitives = build_biometrics_overlay(detection)
        by_key = {primitive.key: primitive for primitive in primitives}

        self.assertEqual(set(by_key), {"face_hud_box", "face_label", "face_baseline"})
        self.assertFalse(
            {"left_eye", "right_eye", "nose", "nose_reticle", "facial_axis", "face_scanline", "status_panel"}
            & set(by_key)
        )
        self.assertEqual(by_key["face_hud_box"].points, (Point(x=10, y=20), Point(x=110, y=140)))
        self.assertEqual(by_key["face_hud_box"].color, NEON_CYAN)
        self.assertEqual(by_key["face_hud_box"].accent_color, NEON_BLUE)
        self.assertEqual(by_key["face_hud_box"].corner_length, 14)
        self.assertEqual(by_key["face_hud_box"].thickness, 2)
        self.assertEqual(by_key["face_label"].points, (Point(x=10, y=14),))
        self.assertEqual(by_key["face_label"].label, "TRACK")
        self.assertEqual(by_key["face_label"].font_scale, 0.45)
        self.assertNotIn("face_label_leader", by_key)
        self.assertEqual(by_key["face_baseline"].points, (Point(x=35, y=146), Point(x=85, y=146)))

    def test_overlay_ignores_missing_eye_data_without_guessing_internal_markers(self):
        detection = DetectedFace(face=Rect(x=100, y=50, width=80, height=100))

        primitives = build_biometrics_overlay(detection)
        by_key = {primitive.key: primitive for primitive in primitives}

        self.assertEqual(set(by_key), {"face_hud_box", "face_label", "face_baseline"})
        self.assertEqual(by_key["face_label"].label, "TRACK")

    def test_classifies_common_hand_gestures_from_finger_state(self):
        self.assertEqual(classify_hand_gesture((1, 1, 1, 1, 1)), "PALM")
        self.assertEqual(classify_hand_gesture((0, 0, 0, 0, 0)), "FIST")
        self.assertEqual(classify_hand_gesture((0, 1, 0, 0, 0)), "POINT")
        self.assertEqual(classify_hand_gesture((0, 1, 1, 0, 0)), "PEACE")

    def test_builds_finger_and_palm_line_primitives_from_landmarks(self):
        landmarks = tuple(
            Point(x=x, y=y)
            for x, y in (
                (60, 150),
                (48, 130),
                (36, 112),
                (24, 98),
                (12, 88),
                (64, 108),
                (62, 82),
                (60, 56),
                (58, 30),
                (82, 104),
                (86, 76),
                (88, 48),
                (90, 20),
                (100, 110),
                (108, 86),
                (114, 62),
                (120, 40),
                (116, 124),
                (132, 106),
                (146, 90),
                (158, 76),
            )
        )
        detection = DetectedHand(
            palm=Rect(x=48, y=104, width=68, height=46),
            fingers_up=(1, 1, 1, 1, 1),
            landmarks=landmarks,
        )

        primitives = build_hand_gesture_overlay(detection)
        by_key = {primitive.key: primitive for primitive in primitives}

        self.assertEqual(
            set(by_key),
            {
                "hand_palm_outline",
                "hand_thumb",
                "hand_index",
                "hand_middle",
                "hand_ring",
                "hand_pinky",
                "hand_gesture_label",
                "hand_center_dot",
            },
        )
        self.assertNotIn("hand_gesture_leader", by_key)
        self.assertNotIn("hand_palm_box", by_key)
        self.assertEqual(
            by_key["hand_palm_outline"].points,
            (landmarks[0], landmarks[1], landmarks[5], landmarks[9], landmarks[13], landmarks[17], landmarks[0]),
        )
        self.assertEqual(by_key["hand_thumb"].points, (landmarks[1], landmarks[2], landmarks[3], landmarks[4]))
        self.assertEqual(by_key["hand_index"].points, (landmarks[5], landmarks[6], landmarks[7], landmarks[8]))
        self.assertEqual(by_key["hand_palm_outline"].color, NEON_MAGENTA)
        self.assertEqual(by_key["hand_index"].color, NEON_CYAN)
        self.assertEqual(by_key["hand_gesture_label"].label, "PALM")
        self.assertEqual(by_key["hand_center_dot"].points, (Point(x=82, y=127),))

    def test_converts_hand_landmarks_to_palm_detection(self):
        landmarks = [[0, 0, 0] for _ in range(21)]
        for index, point in {
            0: [42, 120, 0],
            1: [45, 96, 0],
            5: [50, 78, 0],
            9: [66, 70, 0],
            13: [84, 76, 0],
            17: [92, 108, 0],
        }.items():
            landmarks[index] = point
        hand = {"bbox": (30, 50, 100, 130), "lmList": landmarks}

        detection = _to_detected_hand(hand, (1, 1, 1, 1, 1), overlay=__import__("vision.engine", fromlist=[""]))

        self.assertEqual(detection.palm, Rect(x=42, y=70, width=50, height=50))
        self.assertEqual(detection.landmarks[17], Point(x=92, y=108))
        self.assertEqual(detection.fingers_up, (1, 1, 1, 1, 1))

    def test_renderer_delegates_hud_box_to_cvzone_corner_rect(self):
        fake_cv2 = FakeCV2()
        fake_cvzone = FakeCVZone()
        primitives = (
            OverlayPrimitive(
                key="face_hud_box",
                points=(Point(x=1, y=2), Point(x=11, y=22)),
                color=NEON_CYAN,
                accent_color=NEON_BLUE,
                corner_length=7,
                thickness=2,
            ),
        )

        _draw_overlay(fake_cv2, fake_cvzone, object(), primitives)

        self.assertEqual(fake_cvzone.corner_rects, [((1, 2, 10, 20), 7, 2, 0, NEON_CYAN, NEON_BLUE)])
        self.assertEqual(fake_cv2.rectangles, [])
        self.assertEqual(fake_cv2.lines, [])
        self.assertEqual(fake_cv2.circles, [])

    def test_renderer_draws_hand_lines_without_corner_rectangle(self):
        fake_cv2 = FakeCV2()
        fake_cvzone = FakeCVZone()
        primitives = (
            OverlayPrimitive(
                key="hand_palm_outline",
                points=(Point(x=1, y=2), Point(x=3, y=4), Point(x=5, y=6)),
                color=NEON_MAGENTA,
                thickness=2,
            ),
            OverlayPrimitive(
                key="hand_center_dot",
                points=(Point(x=4, y=5),),
                color=NEON_CYAN,
                radius=3,
                thickness=2,
            ),
        )

        _draw_overlay(fake_cv2, fake_cvzone, object(), primitives)

        self.assertEqual(fake_cvzone.corner_rects, [])
        self.assertEqual(fake_cv2.lines, [((1, 2), (3, 4), NEON_MAGENTA, 2), ((3, 4), (5, 6), NEON_MAGENTA, 2)])
        self.assertEqual(fake_cv2.circles, [((4, 5), 3, NEON_CYAN, 2)])

    def test_renderer_draws_label_text_without_filled_panel(self):
        fake_cv2 = FakeCV2()
        fake_cvzone = FakeCVZone()
        primitives = (
            OverlayPrimitive(
                key="face_label",
                points=(Point(x=8, y=12),),
                color=NEON_CYAN,
                thickness=1,
                label="TRACK",
                font_scale=0.45,
            ),
        )

        _draw_overlay(fake_cv2, fake_cvzone, object(), primitives)

        self.assertEqual(fake_cvzone.text_rects, [])
        self.assertEqual(
            fake_cv2.put_texts,
            [
                ("TRACK", (9, 13), 0, 0.45, (0, 0, 0), 2),
                ("TRACK", (8, 12), 0, 0.45, NEON_CYAN, 1),
            ],
        )

    def test_rejects_invalid_rectangles_before_rendering(self):
        with self.assertRaises(ValueError):
            Rect(x=0, y=0, width=0, height=20)
        with self.assertRaises(ValueError):
            Rect(x=0, y=0, width=20, height=-1)

    def test_cv_dependencies_are_lazy_and_have_clear_install_errors(self):
        try:
            cv2 = require_cv2()
        except OpenCVUnavailableError as error:
            self.assertIn("opencv-python", str(error))
        else:
            self.assertTrue(hasattr(cv2, "VideoCapture"))

        try:
            cvzone = require_cvzone()
        except CVZoneUnavailableError as error:
            self.assertIn("cvzone", str(error))
        else:
            self.assertTrue(hasattr(cvzone, "cornerRect"))



class FakeCV2:
    FONT_HERSHEY_SIMPLEX = 0

    def __init__(self):
        self.rectangles = []
        self.lines = []
        self.circles = []
        self.put_texts = []

    def rectangle(self, frame, start, end, color, thickness):
        self.rectangles.append((start, end, color, thickness))

    def line(self, frame, start, end, color, thickness):
        self.lines.append((start, end, color, thickness))

    def circle(self, frame, center, radius, color, thickness):
        self.circles.append((center, radius, color, thickness))

    def putText(self, frame, text, origin, font, scale, color, thickness):
        self.put_texts.append((text, origin, font, scale, color, thickness))


class FakeCVZone:
    def __init__(self):
        self.text_rects = []
        self.corner_rects = []

    def cornerRect(self, img, bbox, l=30, t=5, rt=1, colorR=(255, 0, 255), colorC=(0, 255, 0)):
        self.corner_rects.append((bbox, l, t, rt, colorR, colorC))
        return img

    def putTextRect(
        self,
        img,
        text,
        pos,
        scale=3,
        thickness=3,
        colorT=(255, 255, 255),
        colorR=(255, 0, 255),
        **kwargs,
    ):
        self.text_rects.append((text, pos, scale, thickness, colorT, colorR))
        return img

if __name__ == "__main__":
    unittest.main()
