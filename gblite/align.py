# -*- coding: utf-8 -*-
"""Whisper 낱말 타임스탬프를 wav2vec2 음소 모델로 다시 잡는다.

**전사를 다시 하지 않는다.** 텍스트는 그대로 두고 시간만 고친다. 그래서
재전사(세션당 17분)와 달리 싸고, 코퍼스 전체에 걸 수 있다.

왜 필요한가 — 2.20절 실측. Whisper 가 낱말을 늘여 잡아 **침묵을 낱말 안으로
삼키고** 있었다. 32분 녹음에서 실제 발화가 17분인데 23.9분으로 잡았고, 그
6.8분이 낱말 경계 안으로 흡수되면서 침묵 마커가 사라졌다.

    김민지 (한국어 93%)   미세침묵  28 → 514      낱말 수 2947 그대로
    참여자E (영어 71%)     발화시간 23.9분 → 17.1분

(주의: 초기에 여기 적었던 "사람 273곳 대 기계 28곳"은 2.20절에서 철회된
계수다 — 서로 다른 양을 나눈 숫자였다. 정렬의 근거는 위의 발화시간 6.8분
흡수와, 정렬 전 위치 오류율 92~100%다. 정렬과 문턱은 짝이다 — 2.24절.)

## 실험 스크립트에서 바뀐 점

`화자검증/try_whisperx.py` 는 간격 분포만 보면 됐으므로 턴을 통째로 정렬해
낱말만 돌려받았다. **거기서 화자가 사라진다.** 산출물에 넣으려면 치명적이라
(화자 분리를 통째로 날린다) 여기서는 화자별로 묶고 되돌려준다.

정렬에 실패한 세그먼트는 **원래 타임스탬프를 유지한다.** 정렬이 안 됐다고
낱말을 버리면 전사가 손상된다 — 시간을 고치려다 내용을 잃는 것은 손해다.
"""
import re
import sys
import unicodedata
from typing import List, Optional, Tuple

from .models import Word

# 공식 정렬 모델 목록에 한국어가 없어 직접 지정한다.
KO_ALIGN_MODEL = "kresnik/wav2vec2-large-xlsr-korean"

MAX_SEG_SEC = 30.0        # 이보다 긴 세그먼트는 쪼갠다 (정렬 실패·메모리)
SPLIT_GAP = 2.0           # 세그먼트 안에 이만한 무음이 있으면 거기서 끊는다

_HAN = re.compile(r"[가-힣]")
_LAT = re.compile(r"[A-Za-z]")
_WARNED = False


# ── 세그먼트 만들기 ──────────────────────────────────────────────────────
def _segments(words: List[Word]) -> List[Tuple[int, int]]:
    """정렬 단위로 쓸 (시작 인덱스, 끝 인덱스+1) 목록.

    화자가 바뀌면 끊는다. 화자를 세그먼트에 묶어 둬야 정렬 뒤에 되돌려줄 수
    있다. 너무 길거나 안에 큰 무음이 있어도 끊는다 — wav2vec2 는 긴 구간에서
    backtrack 이 실패하는 일이 있다.
    """
    out = []
    i = 0
    n = len(words)
    while i < n:
        j = i + 1
        while j < n:
            if words[j].speaker != words[i].speaker:
                break
            if words[j].start - words[j - 1].end >= SPLIT_GAP:
                break
            if words[j].end - words[i].start >= MAX_SEG_SEC:
                break
            j += 1
        out.append((i, j))
        i = j
    return out


def _is_korean(text: str) -> bool:
    return len(_HAN.findall(text)) >= len(_LAT.findall(text))


def _word_key(text: str) -> str:
    """Compare word content while ignoring Unicode form, case, and punctuation."""
    normalized = unicodedata.normalize("NFKC", text or "").casefold()
    return "".join(ch for ch in normalized if ch.isalnum())


# ── 정렬 ────────────────────────────────────────────────────────────────
def align_words(words: List[Word], audio_path: str,
                ko_model: str = KO_ALIGN_MODEL,
                device: str = "cpu") -> List[Word]:
    """낱말 타임스탬프를 다시 잡는다. 실패하면 원본을 그대로 돌려준다."""
    global _WARNED
    if not words or not audio_path:
        return words
    try:
        import whisperx
    except ImportError:
        if not _WARNED:
            _WARNED = True
            print("    [경고] whisperx 가 없어 강제 정렬을 건너뜁니다 — "
                  "침묵 마커가 크게 과소 계산됩니다.\n"
                  "           설치:  pip install whisperx", file=sys.stderr)
        return words

    words = sorted(words, key=lambda w: (w.start, w.end))
    spans = _segments(words)

    # 세그먼트를 언어별로 나눈다.
    #
    # 한국어 모델로 영어를 정렬하면 backtrack 이 조용히 실패해 **원래
    # 타임스탬프로 되돌아간다**(실측: "stop", "Like the video we"). 오류를
    # 내지 않고 그냥 안 고쳐진 채 지나가므로 알아채기 어렵다. 참여자E 세션에서
    # 언어를 갈라 태우자 pause 대 간격이 307 → 597 로 바로잡혔다.
    groups = {"ko": [], "en": []}
    for k, (i, j) in enumerate(spans):
        text = " ".join(w.text for w in words[i:j]).strip()
        if not text:
            continue
        lang = "ko" if _is_korean(text) else "en"
        groups[lang].append((k, {"start": words[i].start,
                                 "end": words[j - 1].end, "text": text}))

    try:
        audio = whisperx.load_audio(audio_path)
    except Exception as e:                                   # noqa: BLE001
        print(f"    [경고] 음원을 열지 못해 정렬을 건너뜁니다: {e}",
              file=sys.stderr)
        return words

    aligned = {}                      # 세그먼트 번호 -> 정렬된 낱말 목록
    for lang, items in groups.items():
        if not items:
            continue
        try:
            model, meta = whisperx.load_align_model(
                language_code=lang, device=device,
                **({"model_name": ko_model} if lang == "ko" else {}))
        except Exception as e:                               # noqa: BLE001
            print(f"    [경고] {lang} 정렬 모델을 불러오지 못했습니다: {e}",
                  file=sys.stderr)
            continue
        try:
            res = whisperx.align([s for _, s in items], model, meta,
                                 audio, device, return_char_alignments=False)
        except Exception as e:                               # noqa: BLE001
            print(f"    [경고] {lang} 정렬 실패: {e}", file=sys.stderr)
            continue
        finally:
            del model
        # whisperx 는 입력 순서를 지켜 돌려준다. 그 순서로 되짚어
        # **원래 세그먼트의 화자**를 물려준다.
        for (k, _), seg in zip(items, res.get("segments", [])):
            got = [w for w in seg.get("words", [])
                   if w.get("start") is not None and w.get("end") is not None]
            expected = spans[k][1] - spans[k][0]
            if got and len(got) != expected:
                # WhisperX가 부분 결과를 반환하면, 반환되지 않은 원본 낱말이
                # 조용히 사라진다. 정렬은 시간 개선 단계이므로 낱말 내용은
                # 건드리지 않고 세그먼트 전체를 원래 시간으로 되돌린다.
                print(f"    [경고] 부분 정렬 결과를 버립니다: 세그먼트 {k} "
                      f"{len(got)}/{expected}개 낱말", file=sys.stderr)
                continue
            if got:
                expected_text = [_word_key(w.text) for w in words[spans[k][0]:spans[k][1]]]
                got_text = [_word_key(w.get("word", "")) for w in got]
                if got_text != expected_text:
                    print(f"    [경고] 정렬 텍스트가 달라 결과를 버립니다: 세그먼트 {k}",
                          file=sys.stderr)
                    continue
            if got:
                aligned[k] = got
        print(f"    정렬 {lang}: 세그먼트 {len(items)}개 중 "
              f"{sum(1 for k, _ in items if k in aligned)}개 성공")

    # ── 되돌려 조립 ──
    out: List[Word] = []
    n_kept = 0
    for k, (i, j) in enumerate(spans):
        speaker = words[i].speaker
        got = aligned.get(k)
        if not got:
            out.extend(words[i:j])            # 실패 — 원본 유지
            n_kept += j - i
            continue
        for w in got:
            out.append(Word(start=float(w["start"]), end=float(w["end"]),
                            text=str(w["word"]).strip(), speaker=speaker,
                            provenance=words[i].provenance,
                            tail_candidate=words[i].tail_candidate))

    out.sort(key=lambda w: (w.start, w.end))
    if n_kept:
        print(f"    정렬 실패 구간의 낱말 {n_kept}개는 원래 시간을 유지했습니다")
    return out


# ── 진단용 ──────────────────────────────────────────────────────────────
def gap_summary(words: List[Word]) -> dict:
    """낱말 사이 간격 분포. 정렬 전후를 비교할 때 쓴다."""
    gaps = [b.start - a.end for a, b in zip(words, words[1:])
            if b.start > a.end]
    return {
        "n_words": len(words),
        "speech_sec": round(sum(w.end - w.start for w in words), 1),
        "micropause": sum(1 for g in gaps if 0.1 <= g <= 0.2),
        "pause": sum(1 for g in gaps if 0.2 < g <= 1.0),
        "large": sum(1 for g in gaps if g > 1.0),
    }
