"""In-memory face recognition over precomputed embeddings."""

from dataclasses import dataclass
import math
from numbers import Real
from typing import Iterable


@dataclass(frozen=True)
class RecognitionResult:
    label: str | None
    score: float
    is_match: bool


class FaceRecognitionEngine:

    def __init__(self, threshold: float = 0.82) -> None:
        if not _is_finite_real(threshold):
            raise ValueError("threshold must be a finite number")
        threshold = float(threshold)
        if threshold < -1.0 or threshold > 1.0:
            raise ValueError("threshold must be between -1.0 and 1.0")

        self._threshold = threshold
        self._templates: list[tuple[str, tuple[float, ...]]] = []
        self._dimension: int | None = None

    def enroll(self, label: str, embedding: Iterable[float]) -> None:
        if not isinstance(label, str) or label == "":
            raise ValueError("label must be a non-empty string")

        values, norm = self._validate_embedding(embedding)
        if self._dimension is None:
            self._dimension = len(values)
        elif len(values) != self._dimension:
            raise ValueError("embedding dimension does not match enrolled templates")

        normalized = tuple(value / norm for value in values)
        self._templates.append((label, normalized))

    def recognize(self, embedding: Iterable[float]) -> RecognitionResult:
        values, norm = self._validate_embedding(embedding)
        if self._dimension is not None and len(values) != self._dimension:
            raise ValueError("embedding dimension does not match enrolled templates")
        if not self._templates:
            return RecognitionResult(label=None, score=0.0, is_match=False)

        best_label: str | None = None
        best_score = -math.inf
        for label, template in self._templates:
            score = math.fsum(template[index] * values[index] for index in range(len(values))) / norm
            if score > best_score:
                best_label = label
                best_score = score

        if best_score >= self._threshold:
            return RecognitionResult(label=best_label, score=best_score, is_match=True)
        return RecognitionResult(label=None, score=best_score, is_match=False)

    def clear(self) -> None:
        self._templates.clear()
        self._dimension = None

    @property
    def labels(self) -> tuple[str, ...]:
        return tuple(label for label, _ in self._templates)

    def __len__(self) -> int:
        return len(self._templates)

    @staticmethod
    def _validate_embedding(embedding: Iterable[float]) -> tuple[tuple[float, ...], float]:
        values: list[float] = []
        norm_squares: list[float] = []
        try:
            iterator = iter(embedding)
        except TypeError as exc:
            raise ValueError("embedding must be an iterable of finite numbers") from exc

        for value in iterator:
            if not _is_finite_real(value):
                raise ValueError("embedding must contain only finite numbers")
            numeric_value = float(value)
            values.append(numeric_value)
            norm_squares.append(numeric_value * numeric_value)

        if not values:
            raise ValueError("embedding must not be empty")

        norm_squared = math.fsum(norm_squares)
        norm = math.sqrt(norm_squared)
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError("embedding must have a finite non-zero norm")

        return tuple(values), norm


def _is_finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)
