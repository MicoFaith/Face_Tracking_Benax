"""
Single-identity speaker recognition helpers (BENAX assessment).

Only the enrolled authorized speaker is identified. All other detected faces
are explicitly ignored and must never drive motor commands.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


class FaceDisposition(Enum):
    AUTHORIZED_SPEAKER = auto()
    IGNORED_OTHER = auto()
    UNKNOWN = auto()


@dataclass
class SpeakerMatchView:
    """Display + control view for one detected face vs the authorized speaker."""

    disposition: FaceDisposition
    label: str
    is_authorized: bool
    confidence: float
    similarity: float
    distance: float

    @property
    def should_track(self) -> bool:
        return self.disposition == FaceDisposition.AUTHORIZED_SPEAKER


def classify_face(
    speaker_name: str,
    matched_name: Optional[str],
    accepted: bool,
    similarity: float,
    distance: float,
) -> SpeakerMatchView:
    """
    Map a cosine-match result to a single-speaker policy label.

    - AUTHORIZED_SPEAKER: exact name match and accepted by threshold
    - IGNORED_OTHER: matched a different enrolled/non-target identity
    - UNKNOWN: no confident match
    """
    sim = float(similarity)
    dist = float(distance)
    conf = sim  # L2-normalized embeddings: similarity is confidence

    if matched_name == speaker_name and accepted:
        return SpeakerMatchView(
            disposition=FaceDisposition.AUTHORIZED_SPEAKER,
            label=speaker_name,
            is_authorized=True,
            confidence=conf,
            similarity=sim,
            distance=dist,
        )

    if accepted and matched_name and matched_name != speaker_name:
        return SpeakerMatchView(
            disposition=FaceDisposition.IGNORED_OTHER,
            label="Ignored",
            is_authorized=False,
            confidence=conf,
            similarity=sim,
            distance=dist,
        )

    return SpeakerMatchView(
        disposition=FaceDisposition.UNKNOWN,
        label="Unknown",
        is_authorized=False,
        confidence=0.0,
        similarity=sim,
        distance=dist,
    )


def smooth_speaker_label(
    history: list[str],
    speaker_name: str,
    raw_label: str,
    window: int = 7,
) -> str:
    """Majority-vote label smoothing biased toward authorized speaker only."""
    history.append(raw_label)
    if len(history) > window:
        history.pop(0)
    if not history:
        return raw_label

    majority = max(set(history), key=history.count)
    unknown_ratio = history.count("Unknown") / len(history)

    if majority == speaker_name and unknown_ratio < 0.5:
        return speaker_name
    if majority == "Ignored":
        return "Ignored"
    return "Unknown"
