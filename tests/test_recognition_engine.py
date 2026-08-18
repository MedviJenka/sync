import math
import unittest
from drone.vision.engine import FaceRecognitionEngine, RecognitionResult


class FaceRecognitionEngineTest(unittest.TestCase):
    def test_recognizes_enrolled_face_by_cosine_similarity(self):
        engine = FaceRecognitionEngine(threshold=0.95)
        engine.enroll("ada", [1.0, 0.0, 0.0])
        engine.enroll("grace", [0.0, 1.0, 0.0])

        result = engine.recognize([0.98, 0.05, 0.0])

        self.assertEqual(result, RecognitionResult(label="ada", score=result.score, is_match=True))
        self.assertGreaterEqual(result.score, 0.95)

    def test_returns_unknown_when_best_score_is_below_threshold(self):
        engine = FaceRecognitionEngine(threshold=0.9)
        engine.enroll("ada", [1.0, 0.0])

        result = engine.recognize([0.0, 1.0])

        self.assertEqual(result.label, None)
        self.assertFalse(result.is_match)
        self.assertAlmostEqual(result.score, 0.0)

    def test_returns_unknown_without_enrolled_templates(self):
        engine = FaceRecognitionEngine()

        result = engine.recognize([1.0, 0.0, 0.0])

        self.assertEqual(result, RecognitionResult(label=None, score=0.0, is_match=False))

    def test_enrollment_copies_and_normalizes_embedding_once(self):
        source_embedding = [3.0, 4.0]
        engine = FaceRecognitionEngine(threshold=0.99)
        engine.enroll("ada", source_embedding)
        source_embedding[0] = 0.0
        source_embedding[1] = 1.0

        result = engine.recognize([0.6, 0.8])

        self.assertEqual(result.label, "ada")
        self.assertTrue(result.is_match)
        self.assertAlmostEqual(result.score, 1.0)

    def test_rejects_invalid_embeddings_and_thresholds(self):
        invalid_embeddings = (
            [],
            [0.0, 0.0],
            [1.0, math.inf],
            [1.0, math.nan],
            ["not-a-number"],
        )

        with self.assertRaises(ValueError):
            FaceRecognitionEngine(threshold=1.5)
        with self.assertRaises(ValueError):
            FaceRecognitionEngine(threshold=-1.5)

        engine = FaceRecognitionEngine()
        with self.assertRaises(ValueError):
            engine.enroll("", [1.0, 0.0])
        for embedding in invalid_embeddings:
            with self.subTest(embedding=embedding):
                with self.assertRaises(ValueError):
                    engine.enroll("ada", embedding)
                with self.assertRaises(ValueError):
                    engine.recognize(embedding)

    def test_rejects_embeddings_with_inconsistent_dimensions(self):
        engine = FaceRecognitionEngine()
        engine.enroll("ada", [1.0, 0.0, 0.0])

        with self.assertRaises(ValueError):
            engine.enroll("grace", [0.0, 1.0])
        with self.assertRaises(ValueError):
            engine.recognize([1.0, 0.0])

    def test_keeps_first_enrolled_identity_on_similarity_tie(self):
        engine = FaceRecognitionEngine(threshold=0.8)
        engine.enroll("ada", [1.0, 0.0])
        engine.enroll("also-ada", [1.0, 0.0])

        result = engine.recognize([1.0, 0.0])

        self.assertEqual(result.label, "ada")
        self.assertEqual(engine.labels, ("ada", "also-ada"))
        self.assertEqual(len(engine), 2)

    def test_clear_removes_all_templates_and_dimension_state(self):
        engine = FaceRecognitionEngine()
        engine.enroll("ada", [1.0, 0.0])
        engine.clear()

        self.assertEqual(len(engine), 0)
        self.assertEqual(engine.labels, ())
        self.assertEqual(engine.recognize([0.0, 1.0, 0.0]), RecognitionResult(label=None, score=0.0, is_match=False))


if __name__ == "__main__":
    unittest.main()
