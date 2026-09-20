# -*- coding: utf-8 -*-
"""
Core data models for GailBot-Lite.

A minimal re-implementation of GailBot's data structures
(word-level tokens and utterances) without the original's
BST/tree machinery.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Word:
    """A single recognized word with timing and speaker label."""
    start: float
    end: float
    text: str
    speaker: str
    # 전사 다중 패스에서 이 낱말이 실제로 채택된 패스. 기존 입력/생성
    # 호출부 호환을 위해 기본값은 빈 문자열로 둔다.
    provenance: str = ""
    # primary의 마지막 낱말 뒤 무한 tail gap에서 채택된 후보인지 여부.
    # 삭제하지 않고 표시만 한다.
    tail_candidate: bool = False

    def to_dict(self) -> dict:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "speaker": self.speaker,
        }


@dataclass
class Marker:
    """A CA marker (pause, gap, overlap boundary, speech-rate delimiter)
    positioned in time. Markers are interleaved with words when rendering."""
    start: float
    end: float
    kind: str            # 'pause' | 'micropause' | 'latch' | 'largepause' |
                         # 'gap' | 'overlap_first_start' | 'overlap_first_end' |
                         # 'overlap_second_start' | 'overlap_second_end' |
                         # 'fast_start' | 'fast_end' | 'slow_start' | 'slow_end'
    info: str = ""       # e.g. duration string
    speaker: str = ""    # speaker the marker attaches to


@dataclass
class Utterance:
    """A sequence of words by the same speaker (a turn-constructional unit)."""
    words: List[Word] = field(default_factory=list)
    markers: List[Marker] = field(default_factory=list)  # markers inside this utt

    @property
    def speaker(self) -> str:
        return self.words[0].speaker if self.words else ""

    @property
    def start(self) -> float:
        return self.words[0].start if self.words else 0.0

    @property
    def end(self) -> float:
        return self.words[-1].end if self.words else 0.0

    @property
    def text(self) -> str:
        return " ".join(w.text for w in self.words)

    # analysis annotations (filled by analysis.SyllableRateAnalyzer)
    syllable_num: Optional[int] = None
    syllable_rate: Optional[float] = None
