# -*- coding: utf-8 -*-
"""웃음(laughter) 검출.

원본 GailBot에는 웃음 분석 모듈이 있으나 이 재구현에는 없었다.
Whisper는 웃음을 받아쓰지 않거나 "하하"로 뭉개므로 그대로 두면 누락된다.

두 경로를 쓴다.
  1) 전사 기반 — "하하", "ㅋㅋ", "haha" 등이 실제로 전사된 경우
  2) 음향 기반 — 사전학습 모델(선택). 설치돼 있으면 자동으로 쓴다.
     omine-me/LaughterSegmentation (Interspeech 2024) 또는
     jrgillick/laughter-detection (Interspeech 2021)

모델이 없으면 1)만 동작하고, 그 사실을 결과에 표시한다.
"""
import re
from typing import List, Optional, Tuple

from .models import Word

# 전사에 실제로 나타나는 웃음 표기
LAUGH_TEXT = re.compile(
    r"^(하하+|허허+|후후+|헤헤+|히히+|크크+|ㅋ{2,}|ㅎ{2,}|"
    r"haha+|hehe+|hoho+|heh+|hah+)$", re.I)


def from_transcript(words: List[Word]) -> List[Tuple[float, float, str]]:
    """전사된 단어 중 웃음 표기를 찾는다."""
    out = []
    for w in words:
        if LAUGH_TEXT.match(w.text.strip()):
            out.append((w.start, w.end, w.speaker))
    return out


def from_audio(audio_path: str,
               min_prob: float = 0.6) -> Optional[List[Tuple[float, float]]]:
    """사전학습 모델로 웃음 구간을 찾는다. 모델이 없으면 None.

    설치 방법 (선택):
        pip install transformers torchaudio
        # omine-me/LaughterSegmentation 가중치는 HuggingFace에서 자동 내려받음
    """
    try:
        from transformers import pipeline
    except ImportError:
        # prosody.extract() 와 같은 방침 — 선택 의존성 부재를 조용히 삼키면
        # 인계받는 사람이 결함을 결과물로 오인한다(개발기록 1.6절 교훈).
        global _WARNED_TRANSFORMERS
        if not globals().get("_WARNED_TRANSFORMERS"):
            globals()["_WARNED_TRANSFORMERS"] = True
            import sys as _sys
            print("    [경고] transformers 가 없어 음향 웃음 탐지를 건너뜁니다 — "
                  "전사 텍스트 기반(ㅋㅋ/haha)만 사용됩니다.\n"
                  "           설치:  pip install transformers torchaudio",
                  file=_sys.stderr)
        return None
    try:
        clf = pipeline("audio-classification",
                       model="omine-me/LaughterSegmentation")
        res = clf(audio_path, chunk_length_s=5, stride_length_s=1)
    except Exception as e:
        # 여기서 조용히 죽고 있었다 (2026-08-13 확인). 위 ImportError 분기에는
        # "조용히 삼키면 결함을 결과물로 오인한다"고 적어 놓고 여덟 줄 아래에서
        # 그 교훈을 어기고 있었다.
        #
        # 실제 원인: omine-me/LaughterSegmentation 저장소에 `config.json` 과
        # `preprocessor_config.json` 이 없어 `pipeline()` 이 OSError 를 낸다.
        # 게다가 그 모델은 audio-classification 이 아니라
        # Wav2Vec2ForAudioFrameClassification 이다(Omine et al. 2024, Interspeech).
        # 가중치(1.26GB)를 수동으로 받아 직접 적재해야 한다.
        #
        # 그래서 사람 전사본에 `(h)` 가 47회인데 기계는 0회였다.
        global _WARNED_MODEL
        if not globals().get("_WARNED_MODEL"):
            globals()["_WARNED_MODEL"] = True
            import sys as _sys
            msg = str(e).split(". Should have")[0][:160]
            print(f"    [경고] 음향 웃음 탐지 모델을 못 불러왔습니다 — 웃음 마커가 "
                  f"하나도 생성되지 않습니다.\n"
                  f"           원인: {type(e).__name__}: {msg}\n"
                  f"           (omine-me/LaughterSegmentation 은 config.json 이 없어 "
                  f"pipeline() 으로 못 씁니다)",
                  file=_sys.stderr)
        return None
    out = []
    for r in res or []:
        if r.get("score", 0) >= min_prob and "laugh" in str(r.get("label", "")).lower():
            ts = r.get("timestamp") or (r.get("start"), r.get("end"))
            if ts and ts[0] is not None:
                out.append((float(ts[0]), float(ts[1])))
    return out


def detect(words: List[Word], audio_path: Optional[str] = None
           ) -> Tuple[List[Tuple[float, float, str]], str]:
    """(웃음 구간, 사용한 방법). 음향 모델이 있으면 합쳐서 돌려준다."""
    hits = from_transcript(words)
    method = "전사 기반"
    if audio_path:
        acoustic = from_audio(audio_path)
        if acoustic:
            # 전사에서 이미 잡힌 구간과 겹치지 않는 것만 추가
            for s, e in acoustic:
                if any(not (e < hs or s > he) for hs, he, _ in hits):
                    continue
                spk = _nearest_speaker(words, (s + e) / 2)
                hits.append((s, e, spk))
            hits.sort()
            method = "전사 + 음향 모델"
    return hits, method


def _nearest_speaker(words: List[Word], t: float) -> str:
    if not words:
        return ""
    return min(words, key=lambda w: abs((w.start + w.end) / 2 - t)).speaker
