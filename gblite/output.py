# -*- coding: utf-8 -*-
"""
Output writers: annotated TXT, CHAT (.cha), CSV (turn & word level),
and utt.toml (GailBot-compatible word-level TOML).

Marker rendering follows CA (Jeffersonian) conventions as used by
GailBot/HiLabSuite:
  micropause              (.)
  pause / large pause     (0.25)      duration in seconds, inside the turn
  latch (same speaker)    =
  latch (turn transition) =  at turn end/next turn start
  cut-off                 word-
  gap                     (1.2)       on its own line between turns
  overlap                 < ... > [<]   /   < ... > [>]
  fast speech             >word word<
  slow speech             <word word>
  sound stretch           wo:rd  (늘어난 정도에 따라 : 1~3개)
  intonation (음높이)     ^ 상승 / ˅ 하강 (Hepburn & Bolden 2013)
  intonation (구두점)     발화 말미에 ? (강한 상승) / , (약한 상승) / . (하강)
  stress                  _word_  (강조 — docx 로 내보낼 때 밑줄)
  volume                  °quiet°  /  LOUD (대문자)
  breath                  .hh
  laughter                (h)
  background noise        ((소음))    전사자 주석 관례인 이중 괄호
"""
import csv
import io
import json
import os
from typing import Dict, List, Tuple

from .models import Word, Marker, Utterance

# 빠름/느림은 Jefferson 표기 `>빠름<` · `<느림>` 을 쓴다.
# 전에는 ∆·∇ 였는데, CA 문헌·CLAN·사람 전사자 모두 부등호를 쓰므로
# 산출물을 사람 전사본과 나란히 놓을 수 없었다.
# (근거: 2026-08-06 교수님 미팅 지시)
#
# 부등호가 겹침 표기(`< ... > [<]`)와 글자를 공유하는 것은 Jefferson 원표기
# 자체의 성질이다. 실제로는 붙임/띄움으로 갈린다 — 속도 기호는 낱말에
# 붙고(`>그러니까<`), 겹침 기호는 띄어 쓴다(`< 그러니까 >`).
FAST_IN = ">"          # 빠름 시작
FAST_OUT = "<"         # 빠름 끝
SLOW_IN = "<"          # 느림 시작
SLOW_OUT = ">"         # 느림 끝
LATCH = "="            # Jefferson latching
CUTOFF = "-"           # 절단(자기 중단) 표시
STRETCH = ":"          # 늘림 — 늘어난 정도에 따라 1~3개
STRESS = "_"           # 강조 (양쪽을 감쌈). docx 밑줄 변환은 이후 단계.
# 음높이의 두드러진 변화. 화살표(↑↓)가 아니라 캐럿(^˅)을 쓴다.
#
# 근거 (2026-08-13):
#   ① 교수님 교재 44쪽 「Transcription Notation」— `^˅ Marked change in
#      pitch: upward or downward.` 출처는 Hepburn & Bolden (2013) 이다.
#      Jefferson 2004 원전은 화살표지만("Arrows indicate shifts into
#      especially high or low pitch"), 교재는 개정판 체계를 따른다.
#   ② 사람 CA 전사본 31개 전수 조사 — `^` 96회(7/31 전사본), `↑` 0회,
#      `↓` 0회, `˅` 0회. 사람은 캐럿만 쓴다.
#   ③ 기호가 달라 억양은 위치 대조를 돌릴 수조차 없었다.
#
# ⚠️ 하강 `˅` 는 사람 전사본에 0회다. 교수님 지시로 넣지만, 정답지에
#    대응물이 없어 **하강의 정밀도는 당분간 잴 수 없다.** 상승과 나눠서
#    보고할 것.
RISE = "^"          # 음높이 상승 (U+005E CIRCUMFLEX ACCENT)
FALL = "˅"     # 음높이 하강 (U+02C5 MODIFIER LETTER DOWN ARROWHEAD)
QUIET = "°"         # 작은 소리 (양쪽을 감쌈)
# 들숨. `h` 개수는 **길이**에 대응한다 — Hepburn & Bolden 2013 §2.5.1
# "the greater the number, the longer the aspiration". Jefferson 2004 도
# ".hhh A dot-prefixed row of h's indicates an inbreath."
#
# 사람 전사본 31개(표 포함) 정확 토큰 실측 (2026-08-13):
#     .h 647 · .hhh 9 · .hhhh 6 · **.hh 0**
# 종전 값 ".hh" 는 사람이 한 번도 안 쓰는 형태라 일치율이 0% 였다.
# 기본형을 `.h` 로 두고, 긴 것은 _breath_hs() 가 h 를 늘린다.
BREATH_IN = ".h"       # 들숨 (기본형)
LAUGH = "(h)"          # 웃음
# 배경 소음. Jefferson 관례에서 이중 괄호는 **전사자의 주석**이고, 소리에
# 대한 설명은 그 안에 적는다(((door slams)) · ((coughs))). 우리는 무엇이
# 났는지까지는 알 수 없으므로 종류를 적지 않고 자리만 표시한다.
NOISE = "((소음))"
OVERLAP_IN = "["       # 겹침 시작 (화자 분리 기반)
OVERLAP_OUT = "]"      # 겹침 끝
NAK = "\x15"      # CLAN time-alignment bullet delimiter
EPS = 1e-3

# 토큰을 옆 낱말에 **붙여** 쓰라는 표시. 조립은 " ".join 인데 `>빠름<` 처럼
# 공백 없이 붙어야 하는 기호가 있어, 붙임 방향을 눈에 안 보이는 제어문자로
# 실어 두었다가 join 직후에 걷어낸다. join 전에 문자열을 만들어 버리면
# 시간순 정렬(events/tokens 병합)을 잃는다.
GLUE_R = "\x00"        # 이 토큰 오른쪽 공백 제거 (여는 기호)
GLUE_L = "\x01"        # 이 토큰 왼쪽 공백 제거 (닫는 기호)

# 억양 구두점 — 발화 맨 끝에 붙는다.
# ⚠️ 억양 윤곽의 표기이지 질문 여부의 판정이 아니다. 개발기록 2.11 —
# 한국어 정중 의문형("~하시겠습니까")은 끝음이 내려가 "." 이 붙는다.
# Jefferson 표기에서 `.` `,` `?` 가 문법 부호가 아니라 억양 기호인 것과 같다.
INTONATION_PUNCT = {
    "intonation_question": "?",    # 강한 상승
    "intonation_continue": ",",    # 약한 상승 — "아직 안 끝났다"
    "intonation_final": ".",       # 하강
}

PAUSE_KINDS = {"pause", "micropause", "latch", "largepause"}


def _unglue(s: str) -> str:
    """GLUE 표시를 걷어내며 그 자리의 공백을 없앤다."""
    return (s.replace(GLUE_R + " ", "").replace(GLUE_R, "")
             .replace(" " + GLUE_L, "").replace(GLUE_L, ""))


def _stretch_colons(info: str) -> int:
    """늘림 배율 → 콜론 개수. info 는 "2.3x" 꼴의 문자열이다.

    2.0~3.0배 → ":"   3.0~4.5배 → "::"   4.5배 초과 → ":::"
    (근거: 2026-08-06 교수님 미팅 지시 — 전에는 2배만 넘으면 일괄 "::" 라
    '조금 늘인 것'과 '아주 길게 끈 것'이 구별되지 않았다.)

    파싱 실패 시 1개로 떨어뜨린다 — 표기를 부풀리는 쪽보다 안전하다."""
    try:
        r = float(str(info).strip().rstrip("xX"))
    except (TypeError, ValueError):
        return 1
    if r > 4.5:
        return 3
    if r >= 3.0:
        return 2
    return 1


def _pause_symbol(m: Marker) -> str:
    """침묵 표기. 길이는 **0.1초 단위**로만 적는다.

    근거 (2026-08-08 CA 데이터 세션 지적): "0.3과 0.32를 귀로 구분하지
    못하면 둘째 자리는 무의미하다." pause 만 소수 둘째 자리로 찍혀
    (0.22) 같은 표기가 나왔는데, Jefferson 관례는 0.1초 단위이고 VAD
    경계 오차가 이미 그보다 크다. micropause 는 종전대로 `(.)` 이다."""
    if m.kind == "micropause":
        return "(.)"
    if m.kind == "latch":
        return LATCH
    return f"({float(m.info):.1f})"


def _speaker_label(speaker: str) -> str:
    # annotate 를 이미 SP_ 가 붙은 utt.toml 에 다시 돌리면 접두사가 누적된다
    # (2026-08-24: 배포 66세션 중 23개가 SP_SP_·SP_SP_SP_ 였다). 이미 붙었으면 안 붙인다.
    return speaker if speaker.startswith("SP_") else f"SP_{speaker}"


# ---------------------------------------------------------------------------
# Inline rendering of one utterance (overlap / fast / slow markers only)
# ---------------------------------------------------------------------------
def _inline(utt: Utterance, markers: List[Marker]) -> str:
    events: List[Tuple[float, int, str]] = []
    final_punct = ""        # 억양 구두점 — 줄 맨 끝에 붙인다
    # 발화 **안**의 침묵을 인라인으로 찍는다(2026-08-12, 2.46절).
    #
    # 전에는 침묵이 발화와 발화 사이에서만 표기됐다. assemble() 의
    # boundary_pause() 가 경계에 딱 붙은 마커만 찾기 때문이다. 그래서 침묵
    # 표기가 TURN_END_THRESHOLD_SECS 에 갇혔다 — 그보다 짧은 침묵은 발화
    # 안에 묻혀 사라졌다. TURN_END 를 0.50 으로 올렸더니 미세침묵이 80개에서
    # 0개가 됐다(CSV 에는 67개가 있는데 전사문에 안 나왔다).
    #
    # 턴 분절과 침묵 표기는 별개 결정이어야 한다. 여기서 발화 안 침묵을
    # 맡으면 TURN_END 를 CA 관행값(0.5~1초)으로 올려도 표기가 살아남는다.
    intra = [m for m in markers
             if m.kind in ("micropause", "pause", "largepause")
             and m.speaker == utt.speaker
             and utt.start - EPS < m.start and m.end < utt.end + EPS]
    for m in intra:
        events.append((m.start, 1, _pause_symbol(m)))
    for m in markers:
        # 들숨은 화자를 특정하지 않고(speaker="") 시간만 갖는다.
        # 화자 필터를 그대로 태우면 영원히 걸러진다.
        if m.kind not in ("breath_in", "noise") and m.speaker != utt.speaker:
            continue
        if not (utt.start - EPS <= m.start <= utt.end + EPS):
            continue
        if m.kind in ("overlap_first_start", "overlap_second_start"):
            events.append((m.start, 0, "<"))
        elif m.kind == "overlap_first_end":
            events.append((m.start, 2, "> [<]"))
        elif m.kind == "overlap_second_end":
            events.append((m.start, 2, "> [>]"))
        elif m.kind == "dz_overlap_start":
            # 화자 분리가 알려준 겹침. 상대방 말은 모노에서 유실돼 한쪽 줄에만
            # 괄호가 붙는다 — '여기서 겹쳤고 그쪽 말은 없다'는 표시다.
            #
            # 정렬 순서가 중요하다. 닫는 괄호의 시각은 마지막 단어의 *끝*이고,
            # 그 값이 다음 단어의 *시작*과 같은 일이 흔하다(단어가 붙어 있을 때).
            # 닫기를 단어(1)보다 뒤에 두면 괄호가 다음 단어를 삼켜
            # `[ 네 이런 날에 [ 그러니까 ] ]` 처럼 중첩된다. 닫기 → 열기 → 단어.
            events.append((m.start, 0.6, OVERLAP_IN))
        elif m.kind == "dz_overlap_end":
            events.append((m.start, 0.5, OVERLAP_OUT))
        elif m.kind == "fast_start":
            events.append((m.start, 0, FAST_IN + GLUE_R))
        elif m.kind == "fast_end":
            events.append((m.start, 2, GLUE_L + FAST_OUT))
        elif m.kind == "slow_start":
            events.append((m.start, 0, SLOW_IN + GLUE_R))
        elif m.kind == "slow_end":
            events.append((m.start, 2, GLUE_L + SLOW_OUT))
        elif m.kind == "intonation_rise":
            # 앞 낱말에 **붙여** 쓴다 — 사람 전사본이 `condition^`,
            # `medical care:^` 처럼 공백 없이 적는다(31개 전수 확인, 96회).
            # 공백을 두면(`네 ^`) 사람 표기와도, 교수님 지시와도 안 맞는다.
            # ※ 교수님 "낱말 앞으로" 지시가 위치 이동을 뜻한다면 GLUE_L 을
            #    GLUE_R 로 바꾸고 정렬 순위를 3 → 0 으로 내리면 된다.
            events.append((m.start, 3, GLUE_L + RISE))
        elif m.kind == "intonation_fall":
            events.append((m.start, 3, GLUE_L + FALL))
        elif m.kind in INTONATION_PUNCT:
            # 발화 **말미**의 억양만 구두점이 된다. 발화 중간에 우연히 같은
            # 시각의 마커가 섞이지 않도록 끝 시각과 맞는 것만 받는다.
            if abs(m.start - utt.end) < EPS:
                final_punct = INTONATION_PUNCT[m.kind]
        elif m.kind == "laughter":
            events.append((m.start, 1, LAUGH))
        elif m.kind == "breath_in":
            # 들숨. 마커는 만들어지고 있었는데 여기서 쓰이질 않아 전사본에
            # 한 번도 나타나지 않았다(사람 전사본은 20~39곳 표기).
            # speaker 가 빈 값이라 위쪽 화자 필터를 타지 않도록 따로 받는다.
            #
            # h 개수는 길이에 대응한다(H&B 2013 §2.5.1). 사람은 `.h` 를
            # 압도적으로 쓰고(647회) 아주 긴 것만 `.hhh`(9)·`.hhhh`(6) 다.
            # 경계값 0.5/0.8초는 **잠정**이다 — 사람 표기와 길이를 짝지어
            # 검증한 적이 없다. PROJECT.md 의 잠정 목록에 있다.
            dur = m.end - m.start
            hs = 3 if dur >= 0.8 else (2 if dur >= 0.5 else 1)
            events.append((m.start, 1, "." + "h" * hs))
        elif m.kind == "noise":
            # 배경 소음. 들숨과 마찬가지로 화자가 없다(speaker=""). 발화
            # **안쪽** 낱말 사이에 떨어진 것만 여기서 찍고, 턴과 턴 사이에
            # 난 소음은 assemble() 이 독립 줄로 낸다.
            events.append((m.start, 1, NOISE))

    cut = {round(m.start, 3) for m in markers
           if m.kind == "cutoff" and m.speaker == utt.speaker}
    stretch = {round(m.start, 3): m.info for m in markers
               if m.kind == "stretch" and m.speaker == utt.speaker}
    quiet = {round(m.start, 3) for m in markers
             if m.kind == "volume_quiet" and m.speaker == utt.speaker}
    loud = {round(m.start, 3) for m in markers
            if m.kind == "volume_loud" and m.speaker == utt.speaker}
    stress = {round(m.start, 3) for m in markers
              if m.kind == "stress" and m.speaker == utt.speaker}

    # 조용한 구간(°...°)은 **낱말 하나씩이 아니라 연속 구간 통째로** 감싼다.
    #   °It's° °often° °a°  →  °It's often a°
    # 근거 (2026-08-08 CA 데이터 세션, Tim Greer 지적): CA 에서 ° 는 조용한
    # **구간**의 여닫이 기호다. 낱말마다 여닫으면 같은 구간을 몇 번씩 여닫는
    # 셈이라 읽는 사람에게는 "온 전사본이 조용하다"로 보인다. 마커 자체는
    # 낱말 단위로 남겨 두고(분석·통계에 필요) 표기만 병합한다.
    quiet_open, quiet_close = set(), set()
    prev_q = False
    for i, w in enumerate(utt.words):
        k = round(w.start, 3)
        is_q = k in quiet
        if is_q and not prev_q:
            quiet_open.add(k)
        if prev_q and not is_q:
            quiet_close.add(round(utt.words[i - 1].start, 3))
        prev_q = is_q
    if prev_q and utt.words:
        quiet_close.add(round(utt.words[-1].start, 3))

    def _render_word(w) -> str:
        key = round(w.start, 3)
        t = w.text
        if key in stretch:              # 늘림: 배율에 비례해 콜론 1~3개
            t = t + STRETCH * _stretch_colons(stretch[key])
        if key in cut:
            t = t + CUTOFF
        if key in loud:
            t = t.upper()
        if key in stress:
            # 강조는 낱말 표기의 바깥에 감싼다. 안쪽 표기(늘림·절단·음량)는
            # 낱말이 어떻게 발음됐는지고, 밑줄은 그 낱말 전체가 강조됐다는
            # 뜻이라 순서를 뒤집으면 `_그러니까_::` 처럼 늘림이 새어 나온다.
            t = f"{STRESS}{t}{STRESS}"
        # ° 는 구간 기호라 낱말 표기보다 더 바깥이다 (`°_It's_ often a°`).
        if key in quiet_open:
            t = QUIET + t
        if key in quiet_close:
            t = t + QUIET
        return t

    tokens: List[Tuple[float, int, str]] = [
        (w.start, 1, _render_word(w)) for w in utt.words]
    tokens += events
    tokens.sort(key=lambda t: (t[0], t[1]))
    return _unglue(" ".join(t[2] for t in tokens)) + final_punct


# ---------------------------------------------------------------------------
# Turn assembly: merge same-speaker utterances joined by a pause marker
# ---------------------------------------------------------------------------
def assemble(utts: List[Utterance], markers: List[Marker]) -> List[dict]:
    """Returns an ordered list of render items:
       {"type": "turn",  "speaker", "text", "start", "end"}
       {"type": "gap",   "info", "start", "end", "prev_speaker"}
       {"type": "noise", "start", "end"}
    Consecutive same-speaker utterances whose boundary carries a
    pause-type marker are merged into one turn with the CA pause symbol
    inline (Jeffersonian style)."""
    pauses = [m for m in markers if m.kind in PAUSE_KINDS]
    gaps = [m for m in markers if m.kind == "gap"]
    latches = [m for m in markers if m.kind == "latch_turn"]
    # 어느 발화 안에도 들어가지 않는 소음 — 턴과 턴 사이에서 난 것이다.
    # _inline() 이 발화 스팬 안의 소음만 가져가므로, 여기서 나머지를 맡지
    # 않으면 조용히 사라진다(들숨이 그렇게 사라지고 있다 — 별건).
    free_noise = [m for m in markers if m.kind == "noise"
                  and not any(u.start - EPS <= m.start <= u.end + EPS
                              for u in utts)]

    def boundary_pause(a: Utterance, b: Utterance):
        for m in pauses:
            if abs(m.start - a.end) < EPS and abs(m.end - b.start) < EPS:
                return m
        return None

    def boundary_gap(a: Utterance, b: Utterance):
        for m in gaps:
            if abs(m.start - a.end) < EPS and abs(m.end - b.start) < EPS:
                return m
        return None

    def boundary_latch(a: Utterance, b: Utterance):
        for m in latches:
            if abs(m.start - a.end) < EPS:
                return m
        return None

    items: List[dict] = []
    prev_utt: Utterance = None
    cur: dict = None
    ni = 0                       # free_noise 소비 위치
    free_noise.sort(key=lambda m: m.start)

    for utt in utts:
        text = _inline(utt, markers)
        # 이 발화가 시작되기 전에 난 소음을 먼저 처리한다.
        due: List[Marker] = []
        while ni < len(free_noise) and free_noise[ni].start < utt.start:
            due.append(free_noise[ni])
            ni += 1
        if cur is not None and prev_utt is not None \
                and cur["speaker"] == utt.speaker:
            m = boundary_pause(prev_utt, utt)
            if m is not None:
                # 같은 화자의 두 발화가 침묵 하나로 이어지는 자리다. 여기서
                # 난 소음은 독립 줄로 빼면 턴이 갈라지므로 침묵 기호 뒤에
                # 이어 붙인다 — `(0.8) ((소음))` 처럼.
                extra = "".join(f" {NOISE}" for _ in due)
                cur["text"] += f" {_pause_symbol(m)}{extra} {text}"
                cur["end"] = utt.end
                prev_utt = utt
                continue
        for m in due:
            items.append({"type": "noise", "start": m.start, "end": m.end})
        # check for a gap between previous utterance and this one
        if prev_utt is not None:
            g = boundary_gap(prev_utt, utt)
            if g is not None:
                items.append({"type": "gap", "info": g.info,
                              "start": g.start, "end": g.end,
                              "prev_speaker": prev_utt.speaker})
        if prev_utt is not None and cur is not None \
                and boundary_latch(prev_utt, utt) is not None:
            cur["text"] += f" {LATCH}"
            text = f"{LATCH} {text}"
        cur = {"type": "turn", "speaker": utt.speaker, "text": text,
               "start": utt.start, "end": utt.end}
        items.append(cur)
        prev_utt = utt
    for m in free_noise[ni:]:            # 마지막 발화 뒤에 남은 소음
        items.append({"type": "noise", "start": m.start, "end": m.end})
    return items


# ---------------------------------------------------------------------------
# Annotated plain-text transcript
# ---------------------------------------------------------------------------
def write_text(path: str, utts: List[Utterance], markers: List[Marker]) -> None:
    with io.open(path, "w", encoding="utf-8") as f:
        for it in assemble(utts, markers):
            if it["type"] == "gap":
                f.write(f"\t({float(it['info']):.1f})\n")
            elif it["type"] == "noise":
                f.write(f"\t{NOISE}\n")
            else:
                f.write(f"{_speaker_label(it['speaker'])}\t{it['text']}"
                        f"  [{it['start']:.2f}–{it['end']:.2f}]\n")


# ---------------------------------------------------------------------------
# CHAT (.cha) — first-draft CLAN-compatible output
# ---------------------------------------------------------------------------
def write_chat(path: str, utts: List[Utterance], markers: List[Marker],
               lang: str = "eng", corpus: str = "gailbot_lite") -> None:
    speakers: List[str] = []
    for u in utts:
        if u.speaker not in speakers:
            speakers.append(u.speaker)
    codes = {s: f"SP{chr(ord('A') + i)}" if i < 26 else f"S{i:02d}"
             for i, s in enumerate(speakers)}

    # --chat-lang 을 안 주면 None 이 그대로 흘러들어와 "@Languages: None" 이
    # 찍혔다(2026-08-13 실측). CHAT 규격에 없는 값이라 CLAN 이 읽지 못한다.
    # 인자 기본값(None)이 함수 기본값("eng")을 덮어쓰는 구조였다 — 함수 쪽에서
    # 막는다. 한/영 혼용 코퍼스이므로 둘 다 적는다(CHAT 다국어 표기 관행).
    if not lang:
        lang = "kor, eng"

    with io.open(path, "w", encoding="utf-8") as f:
        f.write("@Begin\n")
        f.write(f"@Languages:\t{lang}\n")
        f.write("@Participants:\t" + ", ".join(
            f"{c} {_speaker_label(s)} Adult" for s, c in codes.items()) + "\n")
        for c in codes.values():
            f.write(f"@ID:\t{lang}|{corpus}|{c}|||||Adult|||\n")
        prev_code = None
        for it in assemble(utts, markers):
            if it["type"] in ("gap", "noise"):
                code = prev_code or list(codes.values())[0]
                body = (NOISE if it["type"] == "noise"
                        else f"({float(it['info']):.1f})")
                f.write(f"*{code}:\t{body} "
                        f"{NAK}{int(round(it['start']*1000))}_"
                        f"{int(round(it['end']*1000))}{NAK}\n")
            else:
                code = codes[it["speaker"]]
                f.write(f"*{code}:\t{it['text']} "
                        f"{NAK}{int(round(it['start']*1000))}_"
                        f"{int(round(it['end']*1000))}{NAK}\n")
                prev_code = code
        f.write("@End\n")


# ---------------------------------------------------------------------------
# CSV — turn level & word level & markers
# ---------------------------------------------------------------------------
def write_csv_turns(path: str, utts: List[Utterance],
                    markers: List[Marker]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SPEAKER LABEL", "TEXT", "START TIME", "END TIME"])
        for it in assemble(utts, markers):
            if it["type"] == "gap":
                w.writerow(["", f"({float(it['info']):.1f})",
                            f"{it['start']:.2f}", f"{it['end']:.2f}"])
            elif it["type"] == "noise":
                w.writerow(["", NOISE,
                            f"{it['start']:.2f}", f"{it['end']:.2f}"])
            else:
                w.writerow([_speaker_label(it["speaker"]), it["text"],
                            f"{it['start']:.2f}", f"{it['end']:.2f}"])


def write_csv_words(path: str, words: List[Word]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SPEAKER LABEL", "TEXT", "START TIME", "END TIME"])
        for word in sorted(words, key=lambda x: (x.start, x.end)):
            w.writerow([_speaker_label(word.speaker), word.text,
                        f"{word.start:.2f}", f"{word.end:.2f}"])


def write_csv_word_metadata(path: str, words: List[Word]) -> None:
    """Write optional per-word provenance beside the stable GailBot CSV.

    ``*_words.csv`` is consumed by older tools, so its four-column contract is
    intentionally unchanged. This sidecar carries metadata introduced by the
    multi-pass transcription pipeline.
    """
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["SPEAKER LABEL", "TEXT", "START TIME", "END TIME",
                    "PROVENANCE", "TAIL CANDIDATE"])
        for word in sorted(words, key=lambda x: (x.start, x.end)):
            w.writerow([_speaker_label(word.speaker), word.text,
                        f"{word.start:.2f}", f"{word.end:.2f}",
                        word.provenance,
                        "1" if word.tail_candidate else "0"])


def write_csv_markers(path: str, markers: List[Marker]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["TYPE", "INFO", "SPEAKER", "START TIME", "END TIME"])
        for m in markers:
            w.writerow([m.kind, m.info, _speaker_label(m.speaker),
                        f"{m.start:.2f}", f"{m.end:.2f}"])


# ---------------------------------------------------------------------------
# utt.toml — GailBot-compatible word-level TOML
# ---------------------------------------------------------------------------
def write_utt_toml(path: str, words_by_source: Dict[str, List[Word]]) -> None:
    """낱말 단위 TOML. 소스 이름은 **따옴표로 감싼다.**

    감싸지 않으면 TOML 문법이 깨진다. bare key 에는 [A-Za-z0-9_-] 만 쓸 수
    있는데, 우리 소스 이름은 파일 이름에서 오므로 공백·괄호·한글이 섞인다
    (예: `2025-01-17 참여자D (S-02)`). 그렇게 쓴 파일은 tomllib 가
    "Expected ']]' at the end of an array declaration" 로 거부한다 — 즉
    **자기가 쓴 utt.toml 을 자기가 다시 못 읽는** 상태였다. `annotate` 로
    재주석을 돌리려다 발견 (2026-08-06).

    따옴표는 bare key 로도 가능한 이름에 붙여도 무해하다(같은 키로 파싱된다).
    조건 분기 없이 항상 감싸는 편이 안전하다."""
    with io.open(path, "w", encoding="utf-8") as f:
        for source, words in words_by_source.items():
            key = str(source).replace("\\", "\\\\").replace('"', '\\"')
            for word in words:
                text = word.text.replace('"', '\\"')
                f.write(f'[["{key}"]]\n')
                f.write(f"start = {round(word.start, 3)}\n")
                f.write(f"end = {round(word.end, 3)}\n")
                f.write(f'text = "{text}"\n')
                f.write(f'speaker = "{word.speaker}"\n')
                if word.provenance:
                    prov = word.provenance.replace('\\', '\\\\').replace('"', '\\"')
                    f.write(f'provenance = "{prov}"\n')
                if word.tail_candidate:
                    f.write('tail_candidate = true\n')
                f.write('\n')


# ---------------------------------------------------------------------------
# Everything at once
# ---------------------------------------------------------------------------
def write_all(outdir: str, name: str, utts: List[Utterance],
              markers: List[Marker], words_by_source: Dict[str, List[Word]],
              stats: Dict, lang: str = "eng") -> List[str]:
    os.makedirs(outdir, exist_ok=True)
    all_words = [w for ws in words_by_source.values() for w in ws]
    paths = {
        "text": os.path.join(outdir, f"{name}.txt"),
        "chat": os.path.join(outdir, f"{name}.cha"),
        "csv_turn": os.path.join(outdir, f"{name}_turns.csv"),
        "csv_word": os.path.join(outdir, f"{name}_words.csv"),
        "csv_marker": os.path.join(outdir, f"{name}_markers.csv"),
        "toml": os.path.join(outdir, "utt.toml"),
    }
    write_text(paths["text"], utts, markers)
    write_chat(paths["chat"], utts, markers, lang=lang)
    write_csv_turns(paths["csv_turn"], utts, markers)
    write_csv_words(paths["csv_word"], all_words)
    write_csv_markers(paths["csv_marker"], markers)
    write_utt_toml(paths["toml"], words_by_source)
    if stats:
        stats_path = os.path.join(outdir, f"{name}_stats.json")
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        paths["stats"] = stats_path
    return list(paths.values())
