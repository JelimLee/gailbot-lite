# -*- coding: utf-8 -*-
"""
Speaker diarization (who spoke when) for a single mixed recording.

Backend: pyannote.audio 3.x (the CA-research standard).

This module is OPTIONAL. It is only imported when the user passes
`--diarize`, so the base install (faster-whisper only) stays light.
Install the extra with:  bash install_mac.sh --diarize

Flow
----
1. faster-whisper produces word-level timestamps (speaker unknown).
2. pyannote produces speaker turns: (start, end, speaker_label).
3. Each word is relabelled with the speaker whose turn it overlaps most.
"""
import os
import sys
from typing import List, Optional, Tuple

from .models import Word


class DiarizationError(RuntimeError):
    pass


# 화자분리 모델. 환경변수로 갈아끼울 수 있게 둔다 — 모델 비교 실험을 코드
# 수정 없이 하기 위해서다(2026-08-06).
#
#   speaker-diarization-3.1              기본. 로컬 실행, 무료.
#   speaker-diarization-community-1      공개 최신. 로컬 실행, 무료.
#                                        EFlower 시험에서 3.1 과 거의 동일했다.
#   speaker-diarization-precision-2      유료. **가중치가 공개돼 있지 않아
#                                        pyannoteAI 서버로 오디오를 보내야 한다.**
#                                        PYANNOTEAI_API_KEY 가 필요하다.
#
# precision-2 를 쓰려면 녹음이 외부 서버로 나간다. 이 코퍼스는 학생 음성이므로
# 연구 윤리 승인 범위를 먼저 확인해야 한다. 기본값을 로컬 모델로 두는 이유다.
DEFAULT_DIARIZATION_MODEL = "pyannote/speaker-diarization-3.1"


def _model_name() -> str:
    return os.environ.get("GBLITE_DIAR_MODEL") or DEFAULT_DIARIZATION_MODEL


def _load_pipeline(hf_token: Optional[str], device: str,
                   _cache={}):
    # 배치에서 파일마다 재로드하면 파일당 15~20초 순손실 (모델 다운로드 확인
    # + 가중치 로드 + MPS 이동). Whisper 쪽 _model_cache와 같은 방식으로 캐시.
    ck = (hf_token, device, _model_name())
    if ck in _cache:
        return _cache[ck]
    try:
        import torch  # noqa: F401
        from pyannote.audio import Pipeline
    except ImportError:
        raise DiarizationError(
            "화자 분리에는 추가 패키지가 필요합니다.\n"
            "  bash install_mac.sh --diarize\n"
            "를 실행해 pyannote.audio 를 설치하세요."
        )

    token = hf_token or os.environ.get("HF_TOKEN") \
        or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise DiarizationError(
            "HuggingFace 토큰이 필요합니다 (무료, 1회만).\n"
            "  1) https://huggingface.co/settings/tokens 에서 토큰 발급\n"
            "  2) https://huggingface.co/pyannote/speaker-diarization-3.1 및\n"
            "     https://huggingface.co/pyannote/segmentation-3.0 에서\n"
            "     'Agree' 클릭해 약관 동의\n"
            "  3) 실행 시 --hf-token <토큰>  또는  환경변수 HF_TOKEN 설정"
        )

    try:
        # pyannote.audio 4.x renamed the kwarg to `token`; 3.x uses
        # `use_auth_token`. Try the new name first, fall back to the old.
        name = _model_name()
        # precision-2 는 pyannoteAI API 키로 인증한다(HF 토큰이 아니다).
        if "precision" in name:
            token = os.environ.get("PYANNOTEAI_API_KEY") or token
        try:
            pipeline = Pipeline.from_pretrained(name, token=token)
        except TypeError:
            pipeline = Pipeline.from_pretrained(name, use_auth_token=token)
    except Exception as e:  # noqa: BLE001
        raise DiarizationError(
            f"pyannote 모델을 불러오지 못했습니다: {e}\n"
            "토큰이 유효한지, 위 두 모델 약관에 동의했는지 확인하세요."
        )

    # ── 군집 설정 재조정 (2026-08-06 실측) ───────────────────────────────
    # pyannote 기본값은 영어권 회의 데이터로 최적화된 값이다. 이 코퍼스처럼
    # 음색이 가까운 젊은 한국어 화자 집단에서는 과병합이 일어난다.
    #
    # 사람 CA 전사본으로 화자 수 정답을 아는 두 세션에 맞췄다.
    # EFlower 001 = 4명(K·L·S·조교) · EFlower 002 = 3명(L·K·조교)
    #
    #   문턱     하한    001 검출    002 검출
    #   0.7046   12     3명 ✗      3명 ✓      ← 종전 기본값
    #   0.50      4     4명 ✓      5명 ✗      ← 하한을 낮추면 002가 깨진다
    #   0.50     12     4명 ✓      3명 ✓      ← 채택
    #
    # 문턱만 낮춘다. **화자 수를 알려주지 않아도** 001에서 4명을 찾아내고
    # 002를 3명으로 유지한다. 100건 코퍼스는 세션마다 화자 수를 알 수 없으므로
    # 이 성질이 실용적으로 중요하다. 001 기준 경계 재현율도 20.9 → 23.3 이다.
    #
    # 군집 하한(min_cluster_size)은 기본값 12를 유지한다. 4로 낮추면 001은
    # 좋아지지만 002가 3명에서 5명으로 과분할된다. 한 세션에 맞춘 값이
    # 다른 세션을 깨뜨리는 전형이라 채택하지 않는다.
    #
    # 종전 동작으로 되돌리려면 GBLITE_DIAR_THRESHOLD=0.7045654963945799.
    # precision-2(원격)는 파라미터를 서버가 쥐고 있어 적용하지 않는다.
    if "precision" not in _model_name():
        try:
            pipeline.instantiate({
                "segmentation": {"min_duration_off": 0.0},
                "clustering": {
                    "method": "centroid",
                    "min_cluster_size": int(
                        os.environ.get("GBLITE_DIAR_MIN_CLUSTER", 12)),
                    "threshold": float(
                        os.environ.get("GBLITE_DIAR_THRESHOLD", 0.50)),
                },
            })
        except Exception as e:  # noqa: BLE001
            # 모델이 이 파라미터 구조를 안 쓰면 기본값 그대로 간다.
            print(f"    [주의] 군집 설정을 적용하지 못했습니다({e}). 기본값 사용")

    # Prefer Apple Silicon GPU (MPS) when available, else CPU.
    try:
        import torch
        if device == "auto":
            dev = "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            dev = device
        pipeline.to(torch.device(dev))
    except Exception:  # noqa: BLE001
        dev = "cpu"
    _cache[ck] = (pipeline, dev)
    return pipeline, dev


def _load_waveform(audio_path: str, rate: int = 16000):
    """Decode any audio/video file to a mono waveform dict for pyannote.

    pyannote reads files through torchaudio/torchcodec, which needs a system
    ffmpeg. We decode with PyAV (bundled with faster-whisper) instead and hand
    pyannote an in-memory {'waveform', 'sample_rate'} dict, so no system
    ffmpeg is required."""
    import av
    import numpy as np
    import torch

    chunks = []
    with av.open(audio_path) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=rate)
        for frame in container.decode(stream):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))

    if not chunks:
        raise DiarizationError(f"오디오를 읽지 못했습니다: {audio_path}")
    samples = np.concatenate(chunks).astype("float32") / 32768.0
    waveform = torch.from_numpy(samples).unsqueeze(0)  # (channel, time)
    return {"waveform": waveform, "sample_rate": rate}


def diarize_turns(audio_path: str,
                  hf_token: Optional[str] = None,
                  num_speakers: Optional[int] = None,
                  min_speakers: Optional[int] = None,
                  max_speakers: Optional[int] = None,
                  device: str = "auto"
                  ) -> List[Tuple[float, float, str]]:
    """Return a list of (start, end, speaker_label) speaker turns."""
    pipeline, dev = _load_pipeline(hf_token, device)
    print(f"    화자 분리 실행 중 (pyannote, device={dev}) ...")

    kwargs = {}
    if num_speakers:
        kwargs["num_speakers"] = num_speakers
    else:
        if min_speakers:
            kwargs["min_speakers"] = min_speakers
        if max_speakers:
            kwargs["max_speakers"] = max_speakers

    try:
        annotation = pipeline(_load_waveform(audio_path), **kwargs)
    except Exception as e:  # noqa: BLE001
        raise DiarizationError(f"화자 분리 중 오류: {e}")

    # pyannote 4.x returns a DiarizeOutput wrapper; 3.x returns the
    # Annotation directly.
    if not hasattr(annotation, "itertracks"):
        annotation = getattr(annotation, "speaker_diarization", annotation)

    turns: List[Tuple[float, float, str]] = []
    for segment, _, label in annotation.itertracks(yield_label=True):
        turns.append((float(segment.start), float(segment.end), str(label)))
    turns.sort(key=lambda t: (t[0], t[1]))
    return turns


def overlap_spans(turns: List[Tuple[float, float, str]],
                  min_dur: float = 0.10) -> List[Tuple[float, float]]:
    """서로 다른 화자의 turn 이 시간상 겹치는 구간을 뽑는다.

    pyannote 의 segmentation 은 powerset 인코딩이라 '동시에 두 사람'을 직접
    예측하고, 파이프라인 출력에도 겹치는 turn 쌍이 그대로 들어 있다. 그런데
    assign_speakers() 가 단어마다 화자를 하나로 확정하면 단어열이 시간축의
    분할이 되어 두 발화가 겹칠 수 없게 되고, analysis.detect_overlaps() 의
    `nxt.start >= cur.end` 조건이 항상 참이 되어 overlap 마커가 구조적으로
    0이 된다(실측: 통화 1건에서 겹침 11곳이 전부 래치·gap 으로 표기됨).

    그래서 겹침 정보는 **단어 라벨과 무관하게** 여기서 따로 건져 둔다.
    겹친 말의 *내용*은 모노 녹음에서 복원할 수 없지만, *어디서* 겹쳤는지는
    이 구간이 알려준다.
    """
    if not turns:
        return []
    # 시각별 활성 화자 수 변화를 훑는다 (sweep line)
    events: List[Tuple[float, int, str]] = []
    for s, e, lab in turns:
        events.append((s, +1, lab))
        events.append((e, -1, lab))
    events.sort(key=lambda x: (x[0], -x[1]))

    active: dict = {}
    spans: List[Tuple[float, float]] = []
    start: Optional[float] = None
    for t, d, lab in events:
        active[lab] = active.get(lab, 0) + d
        if active[lab] <= 0:
            active.pop(lab, None)
        if len(active) >= 2 and start is None:
            start = t
        elif len(active) < 2 and start is not None:
            if t - start >= min_dur:
                spans.append((start, t))
            start = None

    # 맞닿은 구간 병합
    merged: List[Tuple[float, float]] = []
    for s, e in sorted(spans):
        if merged and s - merged[-1][1] < 0.05:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def _relabel_map(turns: List[Tuple[float, float, str]]) -> dict:
    """Map raw pyannote labels (SPEAKER_00, ...) to compact A, B, C ...
    in order of first appearance."""
    order: List[str] = []
    for _, _, lab in turns:
        if lab not in order:
            order.append(lab)
    mapping = {}
    for i, lab in enumerate(order):
        mapping[lab] = chr(ord("A") + i) if i < 26 else f"S{i:02d}"
    return mapping


def assign_speakers(words: List[Word],
                    turns: List[Tuple[float, float, str]],
                    label_prefix: str = "") -> List[Word]:
    """Relabel each word with the speaker of the turn containing its midpoint.

    종전에는 **겹침이 가장 큰** 구간의 화자를 줬다. 그 방식은 긴 구간에
    구조적으로 유리하다 — 낱말이 긴 구간과 짧은 끼어들기 구간에 걸치면
    겹침의 절대량은 늘 긴 쪽이 크기 때문이다. 짧은 맞장구·끼어들기가
    앞뒤 화자에게 흡수돼 턴 경계가 사라진다.

    실측(2026-08-06, EFlower 001 · 사람-사람 37분 · 사람 전사본 113턴):

        배정 방식        표시 턴    사람 경계 재현율
        최대 겹침(종전)      74            18.6
        중점 소속(현행)     117            20.9

    중점은 구간 길이에 중립이므로 짧은 구간도 자기 낱말을 지킨다. 턴 수가
    사람 전사본 규모(113)와 맞고 경계 재현율도 높다. 챗봇 세션에서는 두
    방식의 차이가 작다 — 합성음과 사람의 구간 길이가 비슷해 편향이 드러나지
    않았다. 사람이 여럿인 녹음에서 비로소 드러났다.

    구간 밖에 놓인 낱말은 중점이 가장 가까운 구간에 붙인다.
    Returns new Word objects (input untouched)."""
    if not turns:
        return words
    mapping = _relabel_map(turns)
    out: List[Word] = []
    for w in words:
        mid = (w.start + w.end) / 2
        best_lab = None
        for (s, e, lab) in turns:
            if s <= mid <= e:
                best_lab = lab
                break
        if best_lab is None:
            # 어느 구간에도 들지 않는 낱말 -> 중점이 가장 가까운 구간
            best_lab = min(
                turns,
                key=lambda t: abs(((t[0] + t[1]) / 2) - mid))[2]
        speaker = f"{label_prefix}{mapping[best_lab]}"
        out.append(Word(start=w.start, end=w.end, text=w.text,
                        speaker=speaker, provenance=w.provenance,
                        tail_candidate=w.tail_candidate))
    return out
