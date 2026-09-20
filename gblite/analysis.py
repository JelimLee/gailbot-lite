# -*- coding: utf-8 -*-
"""
Conversation-analysis annotation logic.

Re-implements the behaviour of GailBot's HiLabSuite analysis plugins
(PausePlugin, GapPlugin, OverlapPlugin, SyllableRatePlugin) on top of
the simplified data model, using the same thresholds as the original
(gb_hilab_suite/src/configs/configData.toml).
"""
import re
import statistics
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .models import Word, Marker, Utterance


# ---------------------------------------------------------------------------
# Thresholds — identical to original GailBot HiLabSuite configData.toml
# ---------------------------------------------------------------------------
@dataclass
class Thresholds:
    GAPS_LB: float = 0.3               # minimum silence between speakers -> gap
    LB_LATCH: float = 0.01             # latch range (same speaker)
    UB_LATCH: float = 0.09
    LB_PAUSE: float = 0.2              # pause range (same speaker)
    UB_PAUSE: float = 1.0
    LB_MICROPAUSE: float = 0.1         # micropause range (same speaker)
    UB_MICROPAUSE: float = 0.2
    LB_LARGE_PAUSE: float = 1.0        # anything >= this is a large pause
    TURN_END_THRESHOLD_SECS: float = 0.1   # words merge into one utterance
                                           # if same speaker & silence < this
    OVERLAP_MARKERLIMIT: float = 4.0   # ⚠️ 죽은 필드 — 원본 configData.toml
                                       # 에서 옮겨왔으나 어디서도 참조 안 함
                                       # (코드 감사 2026-08-06). 원본에서의
                                       # 역할 확인 전까지 제거 보류
    LIMIT_DEVIATIONS: float = 2.0      # fast/slow speech: median +/- 2 * MAD


DEFAULT_THRESHOLDS = Thresholds()


# ---------------------------------------------------------------------------
# 문턱 프리셋 — 기본값을 조용히 바꾸지 않는다
# ---------------------------------------------------------------------------
# 원 논문의 문턱은 **영어 대면 대화**에서 나온 값이고, 논문 스스로 이렇게
# 적어 두었다 — "the standard thresholds for pauses, micropauses and gaps
# may be inappropriate assumptions for some cultures."
#
# 한국어 화상 대화 코퍼스에서 사람 CA 전사본 6건에 맞춰 재보니 현행
# 미세침묵 하한 0.1초는 **확실히 낮다**(2.26절).
#
#     micro 하한   0.10 → 오차 3.320      0.45 → 0.722
#                  0.30 → 오차 2.005      0.50 → 0.610
#
# 다만 0.45~1.5 구간이 거의 평평해 **개수로는 정확한 값을 집을 수 없다.**
# 그래서 기본값을 바꾸지 않고 프리셋으로 둔다. 어느 것을 쓸지는 연구자가
# 정한다 — 기존 산출물과의 비교 가능성이 걸린 문제다.
THRESHOLD_PRESETS = {
    # 원 GailBot HiLabSuite configData.toml 그대로. 기존 산출물과 호환.
    "gailbot": Thresholds(),

    # Gao, Sun & Li (2025), Language Testing. L2 화자 대상으로 100~1000ms 를
    # 훑어 회귀로 고른 값 — 대화 상황에서 청자가 침묵으로 **지각**하는 문턱이
    # 350ms 였다. 문헌 근거가 있는 값.
    # TURN_END_THRESHOLD_SECS 를 LB_MICROPAUSE 와 같게 둔다. 기본값 0.1을
    # 그대로 두면 하한 미만의 침묵(0.1~하한)이 "발화는 갈라지는데 마커는
    # 안 남는" 무표기 분절을 만든다(코드 감사 2026-08-06). 같게 맞추면
    # 하한 미만 침묵은 발화 안에 머물고, 하한 이상은 갈라지며 마커가 붙는다.
    # 마커 출력 자체는 동일하다 — 발화 구조와 렌더링만 달라진다.
    "gao2025": Thresholds(LB_MICROPAUSE=0.35, UB_MICROPAUSE=0.45,
                          LB_PAUSE=0.45, GAPS_LB=0.50,
                          TURN_END_THRESHOLD_SECS=0.35),

    # 우리 코퍼스의 사람 CA 전사본 6건에 맞춘 값. 오차 중앙 1.894 → 0.549.
    # 평평한 바닥의 앞쪽을 잡았다 — 더 올려도 개수 오차는 비슷하지만
    # 마커를 억누르는 쪽이라 신호가 아니다.
    #
    # GAPS_LB 는 격자 최적(1.5)이 아니라 0.5 를 쓴다. 1.5 는 개수 맞추기가
    # "화자 전환 침묵을 아예 안 찍는 쪽"으로 민 억제 잡음이었고(2.26절),
    # 위치 일치율은 0.5 와 1.5 가 동률(67.7 vs 67.8, 2.27절)인데 사람
    # 전사자는 턴 경계 침묵을 찍으므로 이론상 0.5 가 맞다.
    "measured": Thresholds(LB_MICROPAUSE=0.45, UB_MICROPAUSE=0.55,
                           LB_PAUSE=0.55, GAPS_LB=0.50,
                           TURN_END_THRESHOLD_SECS=0.45),

    # ── 정렬된 타임스탬프 전용 (2026-08-12, 2.45절) ──────────────────
    # measured 는 **정렬 전** 타임스탬프(낱말 간격 80~93% 가 0으로 붕괴)에
    # 맞춰 고른 값이다. --align 을 켜면 경계가 조여져 같은 문턱이 조음
    # 간격마다 발화한다(2.24절의 3~54배 과잉). 정렬과 문턱은 짝이다.
    #
    # 이 값들은 "찾은" 것이 아니라 **정한** 것이다. 목적함수를 사람 전사본
    # 개수에 두지 않았다 — 전사자마다 침묵 표기가 14~63개로 갈려(2.24절,
    # EFlower 002 전사자는 1초 미만을 아예 안 적음) 목표가 될 수 없다.
    # 근거는 세 가지다:
    #
    # ⚠ 문턱은 **양자화 격자를 피해** 잡는다. wav2vec2 정렬은 20ms 프레임
    #    단위라 값의 95%가 20ms 배수에 앉는다(0.16·0.30 도 배수다). 문턱을
    #    배수에 두면 그 위에 값이 무더기로 걸터앉아 `>=` 냐 `>` 냐, 부동소수점
    #    오차 하나에 마커가 왔다 갔다 한다. 0.17·0.29 는 근거값(160ms·300ms)을
    #    유지하면서 격자 사이로 반 칸 옮긴 것이다.
    #
    # ① LB_MICROPAUSE=0.17 — CrisperWhisper(Interspeech 2024)의 160ms
    #    조음 간격 상한 + Jefferson 관행(150~200ms). 정렬된 간격 분포에서
    #    20~100ms 에 조음 간격 덩어리(748개 중 60%)가 있고 100ms 이후
    #    급감한다(EFlower 002 실측). wav2vec2 정렬은 20ms 프레임으로
    #    양자화되므로(간격의 95%가 20ms 배수) 160ms 는 그 8배 — 양자화
    #    잡음에 판정이 흔들리지 않는다.
    # ② LB_PAUSE=0.30 — 문헌값이다. De Jong & Bosker(2013)의 250~300ms
    #    (침묵 빈도 지표가 L2 숙련도와 상관 최대인 구간) 상단. 데이터에서
    #    골을 찾으려 했으나 **없었다** — A세션(EFlower 002)의 300~340ms
    #    얕은 골(8건)이 B세션(001, 간격 3,145개)에서 재현되지 않았다
    #    (60건, 이웃 65/54와 평평). 분포는 연속이며, 따라서 이 값의 근거는
    #    데이터가 아니라 문헌이다. 목적함수는 사람 전사본 개수가 아니라
    #    Stivers 2009 FTO 기준선과의 정합 + 문헌값이다(2.45절).
    # ③ TURN_END=0.50 — 1.10-② 의 제약은 "같아야 한다"가 아니라
    #    **"마커 하한보다 작으면 안 된다"** 이다. 작으면 그 사이 침묵이
    #    "발화는 갈라지는데 마커는 안 남는" 무표기 분절을 만든다. 크면
    #    그런 구멍이 없다 — 0.17~0.50 침묵은 마커가 붙고 턴은 이어진다.
    #    처음에 LB_MICROPAUSE 와 같게(0.16) 뒀다가 고쳤다(연구자 지적).
    #    CA 에서 턴 종료는 침묵만으로 정하지 않는다(통사 완결·억양 종결을
    #    함께 본다). 자동 분절에서 관행값은 0.5~1초이고 0.16 은 너무 짧아
    #    0.2초만 쉬어도 턴이 끝난다.
    #
    # GAPS_LB=0.30 은 LB_PAUSE 와 맞춘 결정이다(화자 안/사이에 같은 자로
    # 침묵을 잰다). FTO 분석의 2초 상한은 여기가 아니라 분석 스크립트
    # (multimodal/mm_fto_*)에서 건다 — 마커와 분석은 별개 층이다.
    "aligned": Thresholds(LB_MICROPAUSE=0.17, UB_MICROPAUSE=0.29,
                          LB_PAUSE=0.29, GAPS_LB=0.29,
                          TURN_END_THRESHOLD_SECS=0.50),
}

_HANGUL_RE = re.compile(r"[가-힣]")
_VOWEL_GROUP_RE = re.compile(r"[aeiouy]+", re.IGNORECASE)
_UPPER_RUN_RE = re.compile(r"[A-Z]{2,}")
_DIGIT_RE = re.compile(r"[0-9]")


def estimate_syllables(word: str) -> int:
    """Language-aware syllable estimate.

    - Korean: each Hangul syllable block == one syllable.
    - Latin script: vowel-group heuristic (equivalent in spirit to the
      `syllables` package used by the original).
    - 혼합 토큰("GPT는", "AI가"): 한글과 라틴을 **각각 세어 합산**한다.
      전에는 한글이 하나라도 있으면 라틴을 통째로 무시해 "GPT는"이
      1음절로 잡혔다 — 코드스위칭 코퍼스에서 fast/stretch 마커를
      왜곡시키던 결함(코드 감사 2026-08-06).
    - 숫자·혼합 대소문자 (2026-08-12 추가, 아래 참조).

    **음절을 적게 세면 늘임(`::`)이 늘어난다.** 음절당 시간이
    지속시간÷음절수이므로 분모를 놓치면 그만큼 길어 보인다. 그래서 이
    함수의 오차는 한쪽으로만 위험하고, 애매하면 **많이 세는 쪽**이 안전하다.

    숫자 토큰 (2026-08-12): "2024" 는 모음군이 없어 종전엔 1음절이었는데
    실제 발화는 "twenty twenty-four"(5) · "이천이십사"(5) 다. 숫자 문자당
    1음절로 근사한다 — "2024"→4. 정확하진 않지만 방향이 안전한 쪽(과다
    추정)이고, 자릿수와 음절수가 대체로 함께 늘어난다. 실측에서 숫자 포함
    토큰의 `::` 부착률은 8.0~10.0% 로 세션 기준선(1.4~1.5%)의 6~7배였다.

    혼합 대소문자 (2026-08-12): "ChatGPT" 는 `isupper()` 가 거짓이라 종전엔
    모음군 하나("a")만 잡혀 1음절이었다. 실제로는 "챗·지·피·티"(4)다.
    토큰 **안쪽의 대문자 연속열**(2글자 이상)을 약어로 보고 글자 수만큼 세고,
    나머지만 모음군으로 센다 — "ChatGPT" = GPT(3) + Chat(1) = 4.
    전체 대문자 토큰("GPT")도 이 규칙 하나로 처리된다(연속열 = 토큰 전체).

    ⚠️ 이 두 수정은 **영어 세션의 늘임 과표시(6.11%)를 설명하지 못한다.**
    참여자G 세션의 `::` 232곳을 전수 조사하니 숫자·혼합 토큰이 0곳이었고
    98.7% 가 평범한 1음절 기능어(i · and · um · uh · that's)였다. 실제 원인은
    detect_stretches() 주석에 적었다."""
    hangul_n = len(_HANGUL_RE.findall(word))
    digit_n = len(_DIGIT_RE.findall(word))
    latin = re.sub(r"[^A-Za-z]", "", _HANGUL_RE.sub("", word))
    if not latin:
        return max(1, hangul_n + digit_n)
    # 대문자 연속열은 글자마다 읽는다 — GPT 는 지·피·티, AI 는 에이·아이.
    # 철자 모음군으로 세면 "AI"가 1이 되는데 발음은 2음절이다.
    n = sum(len(a) for a in _UPPER_RUN_RE.findall(latin))
    rest = _UPPER_RUN_RE.sub("", latin)
    if rest:
        m = len(_VOWEL_GROUP_RE.findall(rest))
        # silent trailing 'e' heuristic (라틴 부분에만 적용)
        if rest.lower().endswith("e") and m > 1 and \
                not rest.lower().endswith(("le", "ee")):
            m -= 1
        n += m
    return max(1, hangul_n + digit_n + n)


# ---------------------------------------------------------------------------
# Turn / utterance construction
# ---------------------------------------------------------------------------
def build_utterances(words: List[Word],
                     th: Thresholds = DEFAULT_THRESHOLDS) -> List[Utterance]:
    """Group a time-sorted word list into utterances.

    Words by the same speaker separated by less than
    TURN_END_THRESHOLD_SECS of silence belong to the same utterance
    (same rule as GailBot's UtteranceMapPlugin).
    """
    words = sorted(words, key=lambda w: (w.start, w.end))
    utts: List[Utterance] = []
    # Track the last word of each speaker independently so that
    # cross-talk from another speaker does not split a continuing turn.
    for w in words:
        target: Optional[Utterance] = None
        if utts:
            # find most recent utterance by this speaker
            for u in reversed(utts):
                if u.speaker == w.speaker:
                    fto = w.start - u.end
                    if fto < th.TURN_END_THRESHOLD_SECS:
                        target = u
                    break
        if target is not None:
            target.words.append(w)
        else:
            utts.append(Utterance(words=[w]))
    utts.sort(key=lambda u: (u.start, u.end))
    return utts


# ---------------------------------------------------------------------------
# Pauses (within same speaker) — mirrors PausePlugin
# ---------------------------------------------------------------------------
def detect_intra_pauses(utts: List[Utterance],
                        th: Thresholds = DEFAULT_THRESHOLDS) -> List[Marker]:
    """**발화 안** 낱말 사이 침묵을 찍는다.

    detect_pauses() 는 발화와 발화 **사이**만 본다. 그래서 침묵 마커의
    존재가 TURN_END_THRESHOLD_SECS 에 갇힌다 — 그 값보다 짧은 침묵은
    발화 안에 묻혀 마커를 못 받는다.

    이것이 실제로 터졌다(2026-08-12, 2.46절). TURN_END 를 0.16 → 0.50 으로
    올리자 **미세침묵이 80개에서 0개로 전멸**했다. 0.17~0.50 구간 침묵
    124개가 전부 발화 안으로 들어갔기 때문이다. 턴 과분할을 막으려고
    TURN_END 를 올리면 마커가 사라지고, 마커를 살리려고 내리면 0.2초마다
    턴이 끊기는 딜레마였다.

    두 문제를 분리한다 — **턴 분절은 TURN_END 가, 침묵 표기는 문턱이**
    정한다. 발화 안 침묵은 여기서 인라인으로 찍는다.
    """
    markers: List[Marker] = []
    for u in utts:
        ws = sorted(u.words, key=lambda w: (w.start, w.end))
        for a, b in zip(ws, ws[1:]):
            fto = round(b.start - a.end, 2)
            if fto < th.LB_MICROPAUSE:
                continue
            if fto < th.LB_PAUSE:
                kind = "micropause"
            elif fto <= th.UB_PAUSE:
                kind = "pause"
            elif fto >= th.LB_LARGE_PAUSE:
                kind = "largepause"
            else:
                continue
            markers.append(Marker(a.end, b.start, kind,
                                  info=f"{fto:.1f}", speaker=u.speaker))
    return markers


def detect_pauses(utts: List[Utterance],
                  th: Thresholds = DEFAULT_THRESHOLDS) -> List[Marker]:
    markers: List[Marker] = []
    for cur, nxt in zip(utts, utts[1:]):
        if cur.speaker != nxt.speaker:
            continue
        fto = round(nxt.start - cur.end, 2)
        kind = None
        if th.LB_LATCH <= fto <= th.UB_LATCH:
            kind = "latch"
        elif th.LB_MICROPAUSE <= fto < th.LB_PAUSE:
            kind = "micropause"
        elif th.LB_PAUSE <= fto <= th.UB_PAUSE:
            kind = "pause"
        elif fto >= th.LB_LARGE_PAUSE:
            kind = "largepause"
        if kind:
            # 침묵 길이는 **소수 첫째 자리**까지만 남긴다.
            #
            # 근거 (2026-08-08 CA 데이터 세션 지적): "0.3과 0.32를 귀로
            # 구분하지 못하면 둘째 자리는 무의미하다." Jefferson 관례도
            # 0.1초 단위다. pause 만 `:.2f` 였던 탓에 전사본에 (0.22) 처럼
            # 찍혀, 측정 정밀도를 실제보다 높게 주장하는 표기가 되고 있었다.
            # (VAD 경계 자체의 오차가 이미 수십 ms 이므로 둘째 자리는 잡음이다.)
            # latch 계열의 info 는 렌더링에 쓰이지 않아 그대로 둔다.
            markers.append(Marker(cur.end, nxt.start, kind,
                                  info=f"{fto:.1f}",
                                  speaker=cur.speaker))
    return markers


# ---------------------------------------------------------------------------
# Gaps (between different speakers) — mirrors GapPlugin
# ---------------------------------------------------------------------------
def detect_gaps(utts: List[Utterance],
                th: Thresholds = DEFAULT_THRESHOLDS) -> List[Marker]:
    markers: List[Marker] = []
    for cur, nxt in zip(utts, utts[1:]):
        fto = round(nxt.start - cur.end, 2)
        if fto >= th.GAPS_LB and cur.speaker != nxt.speaker:
            markers.append(Marker(cur.end, nxt.start, "gap",
                                  info=f"{fto:.1f}", speaker=cur.speaker))
    return markers


# ---------------------------------------------------------------------------
# Latching between speakers — Jefferson "="
# ---------------------------------------------------------------------------
def detect_latches(utts: List[Utterance],
                   th: Thresholds = DEFAULT_THRESHOLDS) -> List[Marker]:
    """다음 화자가 간격 없이 곧바로 이어받는 경우(=).

    detect_pauses의 latch는 같은 화자 내부용이므로, CA에서 중요한
    화자 간 래칭은 여기서 따로 잡는다."""
    markers: List[Marker] = []
    for cur, nxt in zip(utts, utts[1:]):
        if cur.speaker == nxt.speaker:
            continue
        fto = round(nxt.start - cur.end, 2)
        if -th.UB_LATCH <= fto <= th.UB_LATCH:
            markers.append(Marker(cur.end, nxt.start, "latch_turn",
                                  info=f"{fto:.2f}", speaker=cur.speaker))
    return markers


# ---------------------------------------------------------------------------
# Cut-off / self-interrupted words — Jefferson "gentri-"
# ---------------------------------------------------------------------------
def detect_stretches(utts: List[Utterance],
                     ratio: float = 2.0,
                     min_syll_sec: float = 0.88) -> List[Marker]:
    """늘림(sound stretch) — Jefferson 표기 `::`

    원본 GailBot에 있으나 이 재구현에 빠져 있던 항목. 음소 정렬 없이도
    단어 지속시간 ÷ 음절 수로 '음절당 시간'을 구해, 같은 화자의 중앙값
    대비 `ratio`배 이상 늘어지고 **동시에** 절대 길이도 min_syll_sec 이상인
    단어를 표시한다.

    화자별 중앙값을 기준으로 삼는 이유는 말 빠르기가 사람마다 다르기
    때문이다. 절대 임계값을 쓰면 느린 화자의 모든 단어가 늘림이 된다.

    min_syll_sec 재보정 (2026-08-11, 지각 문턱 일괄 재보정):
      종전 0.28초는 실측 분포의 p65 언저리여서 사실상 걸러 내는 구실을 하지
      못했고, 낱말의 **14.8%**에 `::` 가 붙었다. 전사본 일곱 낱말에 하나꼴로
      늘림이 표시되면 그건 늘림이 아니라 배경이다.
      실측 (GPT 5세션 · 낱말 20,479, 음절당 시간 초):
        p50=0.20  p90=0.48  p95=0.62  p97=0.76  **p98=0.88**  p99=1.08
      지시대로 상위 약 2% 지점(p98=0.88)을 잡았다. 2배 배율(ratio)은 그 자체가
      지각 가능한 크기이므로 손대지 않는다 — 바뀐 것은 "게다가 절대적으로도
      길어야 한다"는 쪽의 문턱뿐이다.

    ⚠️ 영어 세션 과표시의 원인 (2026-08-12 진단, 수정하지 않고 기록만 한다):
      참여자G 세션만 `::` 가 낱말의 6.11% 였다(다른 두 세션 1.51·1.35%).
      가설은 `estimate_syllables` 의 숫자·혼합 대소문자 과소추정이었는데
      **틀렸다** — 그 세션 232곳을 전수 조사하니 숫자·혼합 토큰이 0곳,
      98.7% 가 평범한 1음절 기능어(i 24 · and 23 · um 18 · uh 13 ·
      that's 13 · but 8 · the 7)였다. 원인은 둘이고 둘 다 이 함수 바깥에 있다.

      ① **낱말 스팬이 뒤따르는 침묵을 삼킨다.** 이 코퍼스의 인접 낱말 간격은
         96.4%(참여자G) · 85.5%(참여자A) · 94.8%(참여자B)가 정확히 0 이다. 즉 침묵이
         앞 낱말의 지속시간에 흡수돼 있다(2.25절과 같은 현상). 실측하면
         `::` 가 붙은 낱말은 스팬의 **28~52%만 소리로 채워져 있고**(전체 낱말은
         97~100%), 침묵을 빼고 다시 재면 음절당 1.08초 → 0.63초로 내려간다.
         지속시간÷음절수는 조음 속도가 아니라 '낱말+뒤 침묵'을 재고 있다.
      ② **영어는 1음절 낱말의 비중이 크다.** 분모가 1이면 흡수된 침묵이
         그대로 음절당 시간이 된다.
           세션        1음절 노출  1음절 안 부착률  다음절 안 부착률  전체
           참여자G    70.1%        8.62%           0.26%        6.11%
           참여자A         28.1%        4.68%           0.28%        1.52%
           참여자B          38.1%        3.28%           0.17%        1.36%
         노출 2.5배 × 부착률 1.8배 ≈ 4배 — 실제 차이(6.11/1.52=4.0배)와 맞는다.

      ③ 남는 부분은 **화자 자신이다.** 침묵을 뺀 유음 시간으로만 봐도
         참여자G의 `::` 낱말은 음절당 중앙 0.63초로 참여자A(0.48)·참여자B(0.33)보다
         길다. 1음절 낱말을 0.6초 넘게 소리 내어 끄는 것은 CA 가 실제로
         표기하는 현상이다. 지표를 '유음 시간'으로 바꾸고 같은 분위로 문턱을
         다시 잡아도 세션 간 순위와 배수는 그대로였다(아래 표) — 즉 6.11% 는
         지표 결함만으로는 설명되지 않는다.

      문턱 스윕 (8세션 32,572낱말, 배율 2.0 고정, 부착률 %):
        절대문턱          0.30   0.40   0.50   0.60   0.70   0.88   1.10
        스팬 기준(현행)   14.18  12.09   8.80   5.86   4.13   2.37   1.09
        유음시간 기준      8.75   6.10   3.46   2.06   1.21   0.55   0.24
      유음시간 기준 0.60초이면 평균 2.06% 로 현행 목표와 같은 자리에 오지만,
      Lu 는 4.96% 로 여전히 한국어 세션(0.65·0.67%)의 7배다. 지표를 바꾸는 것은
      기존 산출물과의 비교 가능성이 걸린 결정이라 여기서 하지 않는다
      (`LIMIT_DEVIATIONS` 를 미뤄 둔 것과 같은 이유, 2.32절)."""
    per_speaker: Dict[str, List[float]] = {}
    word_rate: List[Tuple[Word, str, float]] = []
    for u in utts:
        for w in u.words:
            syll = estimate_syllables(w.text)
            dur = w.end - w.start
            if syll <= 0 or dur <= 0:
                continue
            rate = dur / syll
            per_speaker.setdefault(u.speaker, []).append(rate)
            word_rate.append((w, u.speaker, rate))

    med = {sp: statistics.median(v) for sp, v in per_speaker.items() if v}
    markers: List[Marker] = []
    for w, sp, rate in word_rate:
        base = med.get(sp)
        if base is None:
            continue
        # 중앙값의 ratio배를 넘고, 절대 기준으로도 충분히 길어야 한다.
        # (짧은 감탄사가 통계적으로만 길어 보이는 경우를 배제)
        if rate >= base * ratio and rate >= min_syll_sec:
            markers.append(Marker(w.start, w.end, "stretch",
                                  info=f"{rate / base:.1f}x", speaker=sp))
    return markers


def detect_cutoffs(utts: List[Utterance],
                   th: Thresholds = DEFAULT_THRESHOLDS,
                   max_lookahead: int = 3) -> List[Marker]:
    """말을 하다 끊고 다시 시작한 단어를 표시한다.

    교수님 예시: "this place is gentri- middle class is coming back"
    (gentrified를 말하려다 끊음).

    두 현상을 구분해 표시한다 — 교수님이 절단과 반복을 따로 언급하셨다.
      cutoff     : 뒤에 나오는 더 긴 단어의 앞부분인 경우 → "gentri-"
                   (gentri → gentrified, 그니 → 그니까)
      repetition : 같은 단어를 곧바로 되풀이한 경우 → 표기는 그대로 두고
                   마커만 남긴다 ("I I do agree")

    한계: 말을 끊고 아예 다른 말로 넘어간 경우(gentri- 뒤에 재시도가
    없는 경우)는 Whisper가 온전한 단어로 정규화해버려 탐지되지 않는다."""
    markers: List[Marker] = []
    for u in utts:
        ws = u.words
        for i, w in enumerate(ws):
            a = w.text.strip()
            if not a or a.endswith("-"):
                continue
            for b_word in ws[i + 1:i + 1 + max_lookahead]:
                b = b_word.text.strip()
                if not b:
                    continue
                if a.lower() == b.lower():
                    markers.append(Marker(w.start, w.end, "repetition",
                                          info=a, speaker=u.speaker))
                    break
            # 절단은 바로 다음 단어에서 재시도한 경우만 본다.
            nxt = ws[i + 1] if i + 1 < len(ws) else None
            if nxt is None:
                continue
            b = nxt.text.strip()
            if (_HANGUL_RE.search(a) or not b.lower().startswith(a.lower())
                    or len(a) < 3 or len(b) - len(a) < 2
                    or nxt.start - w.end > 0.25):
                # 한글은 조사·어미 결합이 흔해 접두 일치가 절단의 근거가
                # 되지 못한다(예: "펀드"→"펀드의"). 라틴 문자에 한정하고,
                # 끊김 없이 곧바로 재시도한 경우만 절단으로 본다.
                continue
            markers.append(Marker(w.start, w.end, "cutoff",
                                  info=a, speaker=u.speaker))
    return markers


# 겹침 시작 후 이 시간 안에 발화가 끝나면 '끊겼다'고 본다(초).
OVERLAP_CUTOFF_SEC = 1.0


def detect_overlap_cutoffs(utts: List[Utterance],
                           overlap_spans: Optional[List[Tuple[float, float]]] = None,
                           max_tail: float = OVERLAP_CUTOFF_SEC) -> List[Marker]:
    """겹침에 동반된 절단 — 말하다 끊긴 낱말에 "-" 를 붙인다.

    규칙: 어떤 발화가 진행 중일 때 다른 화자 쪽 겹침이 **시작**되고, 그
    발화가 겹침 시작 후 max_tail 초 안에 끝나면, 그 발화의 마지막 낱말을
    절단으로 표시한다(info="overlap").

    근거 (2026-08-06 교수님 미팅 확정): "말하다 끊기면 오버랩이 되건 말건
    무조건 절단 표시, 대부분 오버랩 진입 직후 끊긴다." 즉 절단의 근거를
    낱말의 **형태**(gentri- 처럼 뒤에 재시도가 있는 경우)가 아니라
    **자리**(겹침 진입 직후 말이 멈춘 자리)에서 찾는다.

    detect_cutoffs() 가 못 잡던 경우를 메운다 — Whisper 는 끊긴 말을 온전한
    낱말로 정규화해 버리므로 형태만 봐서는 절단을 알 수 없다. 반면 겹침
    직후에 발화가 끝났다는 사실은 타임스탬프에 그대로 남는다.

    겹침의 출처는 둘이다:
      1) overlap_spans — 화자 분리(pyannote)가 알려준 구간. diarize 경로는
         낱말마다 화자가 하나로 확정돼 발화가 겹칠 수 없으므로 이쪽뿐이다.
      2) 다른 화자 발화의 시작 시각이 이 발화 안으로 들어온 경우. 채널 분리
         녹음이나 화자별 파일 입력에서 실제로 발화가 겹칠 때 쓰인다.

    한계: 겹침이 시작될 때 상대가 말을 **가로챈** 것인지, 이쪽이 이미 말을
    맺으려던 참이었는지는 시간만으로 구별할 수 없다. 여기서 나온 "-" 는
    '겹침 직후 종료'라는 관찰의 표기이고 절단 의도의 판정이 아니다."""
    onsets: List[Tuple[float, Optional[str]]] = []
    for s, _e in (overlap_spans or []):
        # 화자 분리 구간은 화자를 특정하지 않는다(둘 다 걸쳐 있다).
        onsets.append((s, None))
    for cur, nxt in zip(utts, utts[1:]):
        if cur.speaker != nxt.speaker and nxt.start < cur.end:
            onsets.append((nxt.start, nxt.speaker))
    if not onsets:
        return []
    # **시각으로만** 정렬한다. 그냥 sort() 하면 시각이 같을 때 둘째 원소를
    # 비교하는데, 화자 분리 구간은 화자가 None 이고 발화 전환은 문자열이라
    # `'<' not supported between 'str' and 'NoneType'` 로 죽는다.
    #
    # 잠복해 있다가 2026-08-14 에 터졌다 — `annotate --overlap-from` 으로
    # 이전 산출물의 겹침 구간을 되먹였더니, 그 구간의 시작 시각이 낱말
    # 시작 시각과 정확히 같아 충돌했다. 전사 경로에서는 화자 분리가 준
    # 원본 구간이라 시각이 겹칠 일이 드물어 지금까지 안 터졌다.
    onsets.sort(key=lambda x: x[0])

    markers: List[Marker] = []
    for u in utts:
        if not u.words:
            continue
        for t, src_speaker in onsets:
            if src_speaker == u.speaker:
                continue              # 자기 자신의 겹침 진입은 근거가 못 된다
            # 발화가 '진행 중'이어야 한다 — 시작과 동시에 겹친 경우는 제외.
            if not (u.start < t < u.end):
                continue
            if u.end - t <= max_tail:
                last = u.words[-1]
                markers.append(Marker(last.start, last.end, "cutoff",
                                      info="overlap", speaker=u.speaker))
                break
    return markers


def _dedupe_cutoffs(markers: List[Marker]) -> List[Marker]:
    """같은 낱말에 절단 마커가 두 번 붙는 것을 막는다.

    detect_cutoffs(재시도 기반)와 detect_overlap_cutoffs(겹침 기반)는 서로
    다른 근거로 같은 낱말을 집을 수 있다. 렌더링은 낱말 뒤 "-" 하나라
    화면상 차이는 없지만 CSV 에는 두 줄이 남아 절단 개수가 부풀려진다.
    먼저 들어온 쪽(재시도 기반 — 근거가 더 구체적)을 남긴다."""
    seen = set()
    out: List[Marker] = []
    for m in markers:
        if m.kind == "cutoff":
            key = (round(m.start, 3), m.speaker)
            if key in seen:
                continue
            seen.add(key)
        out.append(m)
    return out


# ---------------------------------------------------------------------------
# Overlaps — mirrors OverlapPlugin
# ---------------------------------------------------------------------------
def detect_overlaps(utts: List[Utterance],
                    th: Thresholds = DEFAULT_THRESHOLDS) -> List[Marker]:
    """For each pair of consecutive utterances that overlap in time and
    belong to different speakers, emit 4 boundary markers delimiting the
    overlapping word stretches in both utterances."""
    markers: List[Marker] = []
    uid = 0
    for cur, nxt in zip(utts, utts[1:]):
        if nxt.start >= cur.end or cur.speaker == nxt.speaker:
            continue
        cw = _overlap_word_span(cur.words, nxt.start, nxt.end)
        nw = _overlap_word_span(nxt.words, cur.start, cur.end)
        if cw is None or nw is None:
            continue
        (c0, c1), (n0, n1) = cw, nw
        markers += [
            Marker(cur.words[c0].start, cur.words[c0].start,
                   "overlap_first_start", str(uid), cur.speaker),
            Marker(cur.words[c1].end, cur.words[c1].end,
                   "overlap_first_end", str(uid), cur.speaker),
            Marker(nxt.words[n0].start, nxt.words[n0].start,
                   "overlap_second_start", str(uid), nxt.speaker),
            Marker(nxt.words[n1].end, nxt.words[n1].end,
                   "overlap_second_end", str(uid), nxt.speaker),
        ]
        uid += 1
    return markers


def detect_diarized_overlaps(utts: List[Utterance],
                             spans: List[Tuple[float, float]]
                             ) -> List[Marker]:
    """화자 분리가 알려준 겹침 구간을 마커로 만든다.

    detect_overlaps() 는 두 발화의 시간이 겹칠 때만 동작한다. 그런데 diarize
    경로에서는 단어마다 화자가 하나로 확정되어 발화가 겹칠 수 없으므로 절대
    발동하지 않는다. 여기서는 `diarize.overlap_spans()` 가 준 구간을 직접 받아
    그 구간에 걸친 단어들을 표시한다.

    한쪽 화자의 말만 전사에 남아 있으므로(모노에서 겹친 말은 복원 불가) 여는
    괄호와 닫는 괄호가 한 줄에만 붙는다. '여기서 상대방과 겹쳤고 그쪽 말은
    유실됐다'는 표시로 읽어야 한다."""
    markers: List[Marker] = []
    for s, e in spans:
        for u in utts:
            idx = [i for i, w in enumerate(u.words)
                   if w.start < e and w.end > s]
            if not idx:
                continue
            markers.append(Marker(u.words[idx[0]].start,
                                  u.words[idx[0]].start,
                                  "dz_overlap_start",
                                  info=f"{e - s:.2f}", speaker=u.speaker))
            markers.append(Marker(u.words[idx[-1]].end, u.words[idx[-1]].end,
                                  "dz_overlap_end",
                                  info=f"{e - s:.2f}", speaker=u.speaker))
    return markers


def _overlap_word_span(words: List[Word], other_start: float,
                       other_end: float) -> Optional[Tuple[int, int]]:
    """Indices (first, last) of words overlapping [other_start, other_end]."""
    idx = [i for i, w in enumerate(words)
           if w.start < other_end and w.end > other_start]
    if not idx:
        return None
    return idx[0], idx[-1]


# ---------------------------------------------------------------------------
# Syllable rate / fast-slow speech — mirrors SyllableRatePlugin
# ---------------------------------------------------------------------------
MIN_UTTS_FOR_RATE = 4     # 화자당 이보다 적으면 빠름/느림을 판정하지 않는다

# 빠름/느림의 **지각 조건**. 화자 중앙값 대비 이 비율 이상 벗어나야 한다.
#
# 왜 필요한가 (2026-08-08 CA 데이터 세션 원칙 — "사람이 귀로 인식할 수 있는
# 차이만 전사에 기록한다"): ±2×MAD 는 그 화자 분포에서의 **통계적 이탈**일
# 뿐이라, 말 빠르기가 고른 화자일수록 MAD 가 작아져 아주 미세한 차이도
# `>빠름<` 이 된다. 속도 변화는 대략 ±25% 이상이라야 들린다.
#
# 실측 (2026-08-11, GPT 5세션 · 발화 2,304): 화자 중앙값 대비 편차 분위 —
#   p5=-51%  p10=-35%  p25=-15%  p50=0%  p75=+18%  p90=+47%  p95=+83%
# 종전 ±2MAD 만으로는 발화의 18.2%(빠름 13.8% + 느림 4.4%)에 부호가 붙었다.
# ±25% 를 AND 로 걸면 지시받은 20~30% 범위의 한가운데이면서, 위 분포에서
# 대략 상위/하위 20% 바깥에 해당한다.
#
#   AND 조건별 부착률 —  ±0%(종전) 18.2% · ±20% 17.8% · **±25% 16.9%** ·
#                        ±30% 15.3% · ±40% 12.2% · ±50% 10.6%
#
# ⚠️ 남은 문제: 이 AND 만으로는 부착률이 크게 줄지 않는다. ±2MAD 를 이미
# 넘은 발화는 대개 ±25% 도 넘기 때문이다 — 즉 지금 과다 표기의 주범은
# MAD 쪽이지 지각 조건 쪽이 아니다. 한 자릿수 %로 내리려면 ±55% 근처가
# 필요한데, 그건 "들리는 차이"라기보다 임의의 절삭이다. 다음 단계는 MAD
# 배수(Thresholds.LIMIT_DEVIATIONS=2.0)를 함께 보는 것이고, 그건 기존
# 산출물과의 비교 가능성 문제라 따로 결정해야 한다.
RATE_MIN_DEV = 0.25


def analyze_syllable_rate(utts: List[Utterance],
                          th: Thresholds = DEFAULT_THRESHOLDS
                          ) -> Tuple[List[Marker], Dict]:
    """빠름/느림 발화 판정 — **화자별** 중앙값·MAD 기준.

    전에는 대화 전체 발화를 한 덩어리로 놓고 중앙값·MAD 를 냈다. 그게
    결함인 이유: 말 빠르기는 사람마다 크게 다르므로, 원래 빠른 화자는
    자기 평소 속도로 말해도 전체 중앙값을 넘어 발화 상당수가 '빠름'으로
    찍히고, 느린 화자는 아무리 빨라져도 절대 '빠름'이 되지 않는다.
    CA 에서 `>빨리<`가 뜻하는 것은 "이 사람 기준으로 갑자기 빨라졌다"이지
    "이 대화 평균보다 빠르다"가 아니다. 같은 파일의 detect_stretches 가
    이미 화자별 중앙값을 쓰고 있었다 — 그쪽 방식에 맞춘다.
    (근거: 2026-08-06 교수님 미팅 지시)

    화자당 발화 MIN_UTTS_FOR_RATE 개 미만이면 판정하지 않는다. 중앙값과
    MAD 는 표본이 두세 개면 그 표본 자신에 끌려다녀 기준선 구실을 못 한다
    (짧게 한마디 거든 화자가 전부 빠름/느림이 되는 것을 막는다).

    2026-08-11: ±2×MAD 에 **지각 조건**(RATE_MIN_DEV)을 AND 로 더했다.
    MAD 기준만으로는 "이 화자 분포에서 드물다"까지만 말할 수 있고 "들린다"는
    말할 수 없다 — 근거는 RATE_MIN_DEV 주석 참조."""
    by_speaker: Dict[str, List[float]] = {}
    for u in utts:
        syll = sum(estimate_syllables(w.text) for w in u.words)
        dur = abs(u.end - u.start) or 0.001
        u.syllable_num = syll
        u.syllable_rate = round(syll / dur, 2)
        by_speaker.setdefault(u.speaker, []).append(u.syllable_rate)

    if not by_speaker:
        return [], {}

    limits: Dict[str, Tuple[float, float]] = {}
    per_speaker_stats: Dict[str, Dict] = {}
    for sp, rates in by_speaker.items():
        if len(rates) < MIN_UTTS_FOR_RATE:
            continue
        med = statistics.median(rates)
        mad = round(statistics.median([abs(r - med) for r in rates]), 2)
        if mad == 0:
            continue          # 분산이 없으면 빠름/느림에 의미가 없다
        upper = med + th.LIMIT_DEVIATIONS * mad
        lower = med - th.LIMIT_DEVIATIONS * mad
        limits[sp] = (lower, upper)
        per_speaker_stats[sp] = {"median": med, "medianAbsDev": mad,
                                 "upperLimit": upper, "lowerLimit": lower,
                                 "utterances": len(rates)}

    markers: List[Marker] = []
    fast = slow = 0
    for u in utts:
        lim = limits.get(u.speaker)
        if lim is None:
            continue
        lower, upper = lim
        # 지각 조건(AND) — 중앙값 대비 ±RATE_MIN_DEV 이상 벗어나야 한다.
        med_sp = per_speaker_stats[u.speaker]["median"]
        dev = abs(u.syllable_rate - med_sp) / med_sp if med_sp else 0.0
        if dev < RATE_MIN_DEV:
            continue
        if u.syllable_rate <= lower and len(u.words) > 1:
            markers.append(Marker(u.start, u.start, "slow_start", speaker=u.speaker))
            markers.append(Marker(u.end, u.end, "slow_end", speaker=u.speaker))
            slow += 1
        elif u.syllable_rate >= upper:
            markers.append(Marker(u.start, u.start, "fast_start", speaker=u.speaker))
            markers.append(Marker(u.end, u.end, "fast_end", speaker=u.speaker))
            fast += 1

    # 전체 통계도 남긴다 — 기존 stats.json 소비자(compare_v1_v2.py 등)가
    # median/medianAbsDev 키를 읽으므로 키 이름은 유지하고, 화자별 값을
    # perSpeaker 로 덧붙인다.
    all_rates = [r for rates in by_speaker.values() for r in rates]
    median = statistics.median(all_rates)
    mad = round(statistics.median([abs(r - median) for r in all_rates]), 2)
    stats = {"median": median, "medianAbsDev": mad,
             "upperLimit": median + th.LIMIT_DEVIATIONS * mad,
             "lowerLimit": median - th.LIMIT_DEVIATIONS * mad,
             "fastturncount": fast, "slowturncount": slow,
             "perSpeaker": per_speaker_stats}
    return markers, stats


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Prosody (intonation / volume / breath) — needs the audio file
# ---------------------------------------------------------------------------
def detect_prosody(utts: List[Utterance], audio_path: str,
                   laugh_spans: Optional[List[Tuple[float, float]]] = None
                   ) -> List[Marker]:
    """억양(↗↘)·음량(° °, 대문자)·호흡(.hh)·배경 소음(((소음))). Praat 기반.

    F0는 화자마다 범위가 달라 절대 Hz로 판정할 수 없다. 말미 기울기를
    반음(semitone)으로 재고, 음량은 화자별 중앙값 대비 상대값으로 본다.

    laugh_spans 는 detect_laughter() 가 잡은 웃음 구간이다. 웃음도 갭에서
    나는 큰 무성음이라 소음으로 이중 표기될 수 있어 미리 빼 준다."""
    from . import prosody
    track = prosody.extract(audio_path)
    if track is None:
        return []

    markers: List[Marker] = []

    # 화자별 발화 스팬 기준 강도. **호흡 검출 전용**으로 남는다 —
    # detect_breaths 는 '낱말 사이 구간이 무음 바닥보다 얼마나 높은가'를 보는
    # 것이라 낱말 단위가 아닌 프레임 중앙값 기준선이 맞다. 음량(°/대문자)
    # 기준선을 낱말 단위로 바꾸면서(아래 base_word_p75) 이 값을 같이 바꾸면
    # 호흡 검출의 대역(base−20 ~ base−5)이 통째로 어긋난다.
    spans: Dict[str, List[Tuple[float, float]]] = {}
    for u in utts:
        spans.setdefault(u.speaker, []).append((u.start, u.end))
    base = {sp: prosody.speaker_db_median(track, sp_spans)
            for sp, sp_spans in spans.items()}
    # 화자별 기준 음높이 — 강조 판정용. 음량과 같은 이유로 화자마다 따로
    # 잡는다 (근거: 2026-08-06 교수님 미팅 지시 — 강조 낱말 표시).
    base_f0 = {sp: prosody.speaker_f0_median(track, sp_spans)
               for sp, sp_spans in spans.items()}
    # 낱말 단위 기준선 두 가지. 둘 다 낱말 스팬에서 내지만 통계량이 다르다.
    #   base_word_db  낱말 **평균**의 중앙값 — 강조(밑줄) 판정용
    #   base_word_p75 낱말 **p75**의 중앙값 — 음량(° °, 대문자) 판정용
    # 음량을 p75 로 옮긴 근거는 prosody.QUIET_DB 주석 참조
    # (2026-08-08 Tim Greer 지적 → 2026-08-11 재보정).
    word_spans: Dict[str, List[Tuple[float, float]]] = {}
    for u in utts:
        for w in u.words:
            word_spans.setdefault(u.speaker, []).append((w.start, w.end))
    base_word_db = {sp: prosody.speaker_word_db_median(track, ws)
                    for sp, ws in word_spans.items()}
    base_word_p75 = {sp: prosody.speaker_word_p75_median(track, ws)
                     for sp, ws in word_spans.items()}

    for u in utts:
        # 억양 — 발화 말미
        #
        # 마커를 두 계열로 낸다 (근거: 2026-08-06 교수님 미팅 지시 —
        # "화살표는 화살표대로 쓰라").
        #   intonation_rise/fall            → ↗ ↘  (기존 표기, 그대로 유지)
        #   intonation_question/continue/final → ? , .  (말미 구두점, 신규)
        # 약한 상승은 화살표를 붙일 만큼 뚜렷하지 않으므로 구두점(",")만 낸다.
        #
        # ⚠️ 구두점은 억양 윤곽 표기이지 질문 여부 판정이 아니다.
        # 개발기록 2.11 — 한국어 정중 의문형은 끝음이 내려간다. 따라서
        # 이 "."이 붙은 발화가 질문이 아니라는 뜻이 **아니다**.
        #
        # base_f0 를 함께 넘겨 말미의 옥타브 오류 프레임을 버린다(2026-08-11).
        # 이 인자가 없으면 피치 추적이 한 옥타브 튄 자리가 그대로 수십
        # 반음/초의 기울기가 되어, 문턱을 지각 수준으로 올려도 화살표가
        # 줄지 않는다 — 근거는 prosody.utterance_slope() 주석 참조.
        slope = prosody.utterance_slope(track, u.start, u.end,
                                        base_f0.get(u.speaker))
        # ── 발화 말미의 `^˅` 는 내지 않는다 (2026-08-13) ──────────────
        #
        # 자리가 틀렸다. Hepburn & Bolden 2013 이 둘을 나눠 놓는다:
        #   §2.3.1 Unit-final intonation → `. , ? ¿ _`  **TCU 말미**
        #   §2.3.3 Pitch variations      → `^ ˅`        **발화 중간**
        #     "Pitch variations can be marked within a word ... or across a
        #      string of words"  예: `Can ↑we pl'se bring↑ the matt:↓ress:.`
        # 교수님 육성(2026-08-13 회의)도 같다 —
        #   "이거는 항상 마지막에만 나오는 거고"(구두점)
        #   "중간에 내가 이렇게 올리는 거야. 단어 사이에 단어 중간에 끼어요"(^)
        #   "지금 퀘스천 마크하고 올라갔잖아. 같이 있잖아요"  ← 이 중복 지적
        #
        # 종전에는 화살표와 구두점을 **둘 다 u.end** 에 찍어 `맞아요^?` 가 됐다.
        #
        # 중간 도약 검출은 아직 없다. 두 가지 방식으로 재봤으나 실패했다
        # (2026-08-13, tools/caret_delta.py · tools/caret_register.py):
        #   이웃 낱말 Δ · 발화 안 리셋 · 레지스터(3초) · 화자 전체 대비
        #   → 사람 `^` 자리와 기준선의 배율이 넷 다 **2배를 못 넘었다**.
        #   정답지도 얇다 — 8세션에 38개, 그중 15개만 정렬된다(4/8 세션 0회).
        # 검출이 서기 전까지 이 마커는 나오지 않는다. 그것이 Jefferson 2004 의
        # "absence of an utterance-final punctuation marker indicates some sort
        #  of 'indeterminate contour'" 와 어긋나지 않는다.
        #
        # `prosody.intonation_arrow()` 함수 자체는 남긴다. 지금은 호출부가
        # 없지만(2026-08-13 확인) 중간 도약 검출이 서면 등급 판정에 다시 쓴다.
        # diag/intonation_*.py 는 `utterance_intonation()` 쪽을 쓰므로 무관하다.
        punct = {"rise": "question", "weak_rise": "continue",
                 "fall": "final"}.get(prosody.intonation_grade(slope))
        if punct:
            markers.append(Marker(u.end, u.end, f"intonation_{punct}",
                                  info=f"{slope:.1f}", speaker=u.speaker))
        # 음량 — 단어 단위
        bf = base_f0.get(u.speaker)
        bwd = base_word_db.get(u.speaker)
        bwp = base_word_p75.get(u.speaker)
        stressed = 0
        # quiet 는 **연속 2낱말 이상**일 때만 마커를 낸다.
        #
        # 근거 (2026-08-08 CA 데이터 세션, Tim Greer 지적): CA 의 °...° 는
        # "주변보다 눈에 띄게 조용한 **구간**"을 뜻하지, 약화된 낱말 하나를
        # 표시하는 기호가 아니다. 영어 기능어(of · the · and)는 원래 약화되어
        # 발음되므로 낱말 단위로 문턱을 넘는 일이 잦은데, 그걸 하나씩 표기하면
        # 전사본 전체에 ° 가 흩뿌려져 "녹음이 잘못된 것 아니냐"는 인상을 준다.
        #
        # 실측 (2026-08-11, 5세션 20,432 낱말): quiet 판정 낱말이 이루는 run
        # 2,598개 중 **93.5%가 1낱말짜리 고립**이었다. 고립을 버리면 산발
        # 오탐이 사라지면서 진짜 '조용한 구간'만 남는다.
        #
        # loud(대문자)는 종전대로 낱말 단위다 — 한 낱말만 크게 지르는 것은
        # CA 에서 실제로 일어나고 관례상 그렇게 표기한다.
        vols = [prosody.word_volume(track, w.start, w.end, bwp)
                for w in u.words]
        for i, (w, vol) in enumerate(zip(u.words, vols)):
            if vol == "quiet":
                prev_q = i > 0 and vols[i - 1] == "quiet"
                next_q = i + 1 < len(vols) and vols[i + 1] == "quiet"
                if not (prev_q or next_q):
                    vol = None
            if vol:
                markers.append(Marker(w.start, w.end, f"volume_{vol}",
                                      info=w.text, speaker=u.speaker))
            # 강조 — 발화당 최대 2개, 한 글자 낱말 제외.
            # 상한을 두는 이유: 화자가 통째로 흥분해 목소리를 높인 구간이면
            # 낱말이 전부 기준선을 넘어 발화 전체에 밑줄이 그어진다. 그러면
            # '어디가 강조인지'라는 정보가 사라진다. 한 글자 낱말("네", "아")은
            # 원래 짧고 세게 나오는 일이 잦아 기준선을 쉽게 넘는다.
            if stressed < 2 and len(w.text.strip()) > 1:
                st = prosody.word_stress(track, w.start, w.end, bwd, bf)
                if st:
                    markers.append(Marker(w.start, w.end, "stress",
                                          info=st, speaker=u.speaker))
                    stressed += 1

    # 호흡 — 단어 사이 구간
    gaps: List[Tuple[float, float]] = []
    for u in utts:
        for a, c in zip(u.words, u.words[1:]):
            if c.start - a.end > 0:
                gaps.append((a.end, c.start))
    for u_prev, u_next in zip(utts, utts[1:]):
        if u_next.start > u_prev.end:
            gaps.append((u_prev.end, u_next.start))
    overall = statistics.median([v for v in base.values() if v]) \
        if any(base.values()) else None
    # info 에 검출 근거(길이·기준선 대비 dB·무게중심·포락선)를 싣는다.
    # 비어 있으면 사후 청취 검증에서 "왜 여기가 찍혔는가"를 되짚을 수 없다 —
    # 2026-08-12 참여자A 46건 검증에서 실제로 막혔던 지점이다.
    for s, e, info in prosody.detect_breaths(track, gaps, overall):
        markers.append(Marker(s, e, "breath_in", info=info, speaker=""))

    # 배경 소음 — 들숨과 같은 기준선을 쓰되 반대쪽 대역이다(prosody.NOISE_DB).
    #
    # 갭의 정의가 들숨과 다르다. 들숨은 위에서 만든 gaps(같은 발화 안 낱말
    # 사이 + 이웃한 두 발화 사이)를 쓰지만, 소음은 **어느 화자의 낱말 구간에도
    # 속하지 않는** 구간이어야 한다. 겹침이 있으면 발화 사이 갭이라도 다른
    # 화자가 그 자리에서 말하고 있을 수 있고, 그 말소리를 소음으로 적으면
    # 안 된다. 그래서 낱말 구간 합집합의 여집합을 따로 낸다.
    spans_all = sorted((w.start, w.end) for u in utts for w in u.words)
    merged: List[List[float]] = []
    for s, e in spans_all:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    free = [(a[1], b[0]) for a, b in zip(merged, merged[1:]) if b[0] > a[1]]
    for s, e, info in prosody.detect_noise(track, free, overall, laugh_spans):
        markers.append(Marker(s, e, "noise", info=info, speaker=""))

    return markers


# ---------------------------------------------------------------------------
# Laughter — original GailBot had this module; this port did not
# ---------------------------------------------------------------------------
def detect_laughter(utts: List[Utterance], words: List[Word],
                    audio_path: Optional[str] = None) -> List[Marker]:
    from . import laughter
    hits, method = laughter.detect(words, audio_path)
    return [Marker(s, e, "laughter", info=method, speaker=sp)
            for s, e, sp in hits]


def analyze(words: List[Word],
            th: Thresholds = DEFAULT_THRESHOLDS,
            audio_path: Optional[str] = None,
            overlap_spans: Optional[List[Tuple[float, float]]] = None
            ) -> Tuple[List[Utterance], List[Marker], Dict]:
    """words -> (utterances, all CA markers, conversation stats)

    audio_path를 주면 운율(억양·음량·호흡)과 웃음까지 분석한다.
    없으면 시간값만으로 가능한 마커만 만든다.

    overlap_spans 는 화자 분리가 알려준 겹침 구간이다(diarize.overlap_spans()).
    diarize 경로에서는 단어마다 화자가 하나로 확정돼 발화가 겹칠 수 없으므로
    detect_overlaps() 가 절대 발동하지 않는다. 이 인자를 주면 그 정보를 살린다."""
    utts = build_utterances(words, th)
    markers: List[Marker] = []
    markers += detect_pauses(utts, th)
    markers += detect_intra_pauses(utts, th)
    markers += detect_gaps(utts, th)
    markers += detect_latches(utts, th)
    markers += detect_cutoffs(utts, th)
    # 겹침 기반 절단은 재시도 기반 **다음**에 와야 한다 — 겹치면 먼저 들어온
    # 쪽을 남기는 규칙이고, 재시도 기반이 근거가 더 구체적이다.
    markers += detect_overlap_cutoffs(utts, overlap_spans)
    markers = _dedupe_cutoffs(markers)
    markers += detect_stretches(utts)
    markers += detect_overlaps(utts, th)
    if overlap_spans:
        markers += detect_diarized_overlaps(utts, overlap_spans)
    if audio_path:
        # 웃음이 먼저다 — detect_prosody 의 소음 판정이 웃음 구간을 제외하려면
        # 그 구간을 이미 알고 있어야 한다.
        laughs = detect_laughter(utts, words, audio_path)
        markers += laughs
        markers += detect_prosody(utts, audio_path,
                                  [(m.start, m.end) for m in laughs])
    rate_markers, stats = analyze_syllable_rate(utts, th)
    markers += rate_markers
    markers.sort(key=lambda m: (m.start, m.end))
    return utts, markers, stats
