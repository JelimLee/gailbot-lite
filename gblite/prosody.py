# -*- coding: utf-8 -*-
"""운율(prosody) 분석 — 억양·음량·호흡.

Praat(Parselmouth)로 F0·강도를 뽑아 Jefferson 표기에 필요한 값을 만든다.
  · 억양: 발화 말미 F0 기울기 → 상승(↗)·하강(↘)·평탄
  · 음량: 주변 대비 강도 → 작게(° °)·크게(대문자)
  · 호흡: 단어 사이 저강도·무성 구간 → 들숨(.hh)

F0는 화자마다 범위가 다르므로 **화자별 중앙값 기준 상대값**으로 판정한다.
절대 Hz로 판정하면 성별·개인차에 그대로 휘둘린다.
"""
import bisect
import os
import sys
from typing import Dict, List, Optional, Tuple

# 화자별 F0 탐색 범위 (Hz). 넓게 잡고 실제 분포에서 중앙값을 취한다.
PITCH_FLOOR = 60.0
PITCH_CEIL = 500.0

# 판정 임계값
TAIL_SEC = 0.35      # 말미로 볼 구간 길이
BREATH_MIN = 0.20    # 호흡으로 볼 최소 길이(초). 2026-08-12: 0.15 → 0.20
BREATH_MAX = 1.20    # 이보다 길면 그냥 침묵으로 본다
# 들숨의 강도 대역 — 화자 기준선보다 CEIL 이상 낮고 FLOOR 이하로만 낮을 것.
# 값 선택 근거는 detect_breaths() 주석(2026-08-11 재보정) 참조.
BREATH_DB_CEIL = 8.0
BREATH_DB_FLOOR = 12.0
# 들숨 판별 게이트 (2026-08-12 신설). 연구자 청취 판정 46건이 근거이고
# 성적표는 detect_breaths() 주석에 있다.
BREATH_COG_MAX = 3000.0   # 스펙트럼 무게중심 상한(Hz)
BREATH_ENV_MIN = 6.0      # 포락선(구간 피크dB − 중앙dB) 하한
BREATH_ENV_MAX = 14.0     # 〃 상한

# ── 배경 소음 (((소음))) ───────────────────────────────────────────────
# 들숨과 **같은 갭**을 보되 반대쪽 끝의 대역을 쓴다. 들숨은 기준선보다
# 8~12dB 낮고, 소음은 3dB 아래까지만 낮다 — 사이에 5dB 의 빈 띠가 있어
# 한 구간이 두 마커를 동시에 받는 일이 없다. 값 근거는 detect_noise() 주석.
NOISE_DB = -3.0          # 기준선 대비 이 이상이면 '발화 수준에 근접'
NOISE_MIN = 0.20         # 여백을 뺀 '속'의 최소 길이(초)
NOISE_MAX = 5.0          # 이보다 길면 소음 사건이 아니라 구간이다
NOISE_EDGE = 0.08        # 갭 양 끝에서 잘라 내는 여백(초)
NOISE_VOICED_FRAC = 0.10  # 유성 프레임 비율 상한
NOISE_MID_DB = -6.0      # 중대역(500~4000Hz) 비중 하한 = '광대역일 것'
MID_BAND_LO = 500.0      # 중대역 통과 필터
MID_BAND_HI = 4000.0
MID_BAND_SMOOTH = 100.0  # Hann 가장자리 폭(Hz)

# 억양 문턱 — 단위는 반음/초(st/s)지만 **고르는 기준은 말미 변화량**이다.
#
# 변화량(excursion) = 기울기 × TAIL_SEC. 즉 "말미 0.35초 동안 음높이가 몇
# 반음 움직였나"이고, 사람이 듣는 것은 기울기가 아니라 이 변화량이다.
#
# 배경 (2026-08-08 CA 데이터 세션 지적 → 2026-08-11 재보정): 종전
# RISE_ST=1.5 st/s 는 변화량으로 **0.5반음**, 쉼표 문턱(0.4배)은 **0.2반음**
# 이었다. 음높이 움직임의 지각 하한은 문맥상 대략 1.5~2반음이므로, 둘 다
# 사람이 들을 수 없는 크기를 표기하고 있었다. 아래 QUESTION_ST 주석에
# 이미 "이 값은 사실상 부호 판정"이라고 적혀 있던 것이 같은 현상이다.
#
# 실측 (2026-08-11, GPT 5세션 · 발화 2,304 중 말미 기울기 산출 1,569):
#   |기울기| 분위 — p10=1.7  p25=5.0  p50=11.8  p75=26.2  p90=49.3 (st/s)
#   변화량 환산   — p10=0.6  p25=1.75 p50=4.1   p75=9.2   p90=17.3 (반음)
#   변화량 문턱별 화살표 부착률(전체 발화 기준)
#     0.5반음(종전) 62.3% · 1.0반음 57.9% · **1.5반음 53.4%** ·
#     2.0반음 48.7% · 3.0반음 41.4%
#
# 1.5반음(=4.29 st/s)을 골랐다. 지시받은 범위(1.5~2.0)의 아래끝이고,
# 실측 p25(1.75반음)와 같은 자리라 "하위 4분의 1은 지각 하한 미만"이라는
# 한 문장으로 설명된다. 화살표·쉼표·마침표 모두 이 하나의 지각 바닥을
# 쓰고, 그 위에서 "?"만 QUESTION_ST(=3.5반음)로 갈린다.
RISE_ST = 4.29       # st/s. 말미 변화량 +1.5반음 (= 1.5 / TAIL_SEC)
FALL_ST = -4.29      # st/s. 말미 변화량 -1.5반음

# 약한 상승(",")의 하한 — RISE_ST 의 몇 배부터 '약한 상승'으로 볼지.
# (근거: 2026-08-06 교수님 미팅 지시 — 억양을 . , ? 3등급으로)
# 기존에는 RISE_ST 미만이 전부 '평탄(None)'으로 버려졌는데, CA 에서
# 말끝을 살짝 올려 두는 것은 "아직 안 끝났다"는 계속 신호이고 사람
# 전사자는 그 자리에 쉼표를 찍는다.
#
# 2026-08-11: 0.4 → 1.0. 종전에는 화살표 문턱의 40%(0.2반음) 부터 쉼표를
# 찍었는데 그 크기는 들리지 않는다. 이제 쉼표의 하한은 화살표와 같은
# 지각 바닥(1.5반음)이고, 위쪽은 종전대로 "?" 문턱에서 끊긴다. 즉
# ',' 는 "들릴 만큼 올렸지만 질문만큼은 아닌" 구간을 뜻한다.
# (실측: 이 변경으로 ',' 부착률 12.2% → 6.6% of 발화)
WEAK_RISE_FRAC = 1.0

# '?'(강한 상승)의 문턱. 지시대로 RISE_ST 와 같게 두어, 화살표 ↗ 가 붙는
# 자리에 '?' 도 붙는다.
#
# ⚠️ 실측 경고 (S-02, 발화 230개 중 기울기 산출 149개):
#   말미 기울기 분위 —  p10=-25.7  p25=-9.9  p50=0.0  p75=+9.9  p90=+22.9
#   구간별 개수      —  <=-1.5: 67개 · -1.5~0.6: 7개 · 0.6~1.5: 2개 · >=1.5: 73개
# 기울기의 실제 산포가 ±40반음/초인데 문턱이 1.5 라, 이 값은 사실상
# **부호 판정**이다. 그래서 3등급이 아니라 2등급처럼 동작한다 —
# 약한 상승(",")이 230개 발화 중 2개밖에 안 나왔고, "?" 는 32%에 붙었다.
# 질문이 발화의 3분의 1일 리는 없다.
#
# 그래서 10.0 으로 올렸다. 위 분포에서 "?" 37개 · "," 36개 · "." 67개로
# 갈린다 — 교수님이 요구한 "이 세 개만 구별하면 된다"가 실제로 성립하는
# 유일한 지점이다. RISE_ST(=1.5)는 건드리지 않았으므로 화살표 ↗↘ 는
# 종전 그대로 붙는다. "화살표는 화살표대로" 지시가 유지된다.
#
# 이 값은 S-02 한 세션의 분포로 정했다. 세션이 늘면 재확인이 필요하고,
# 되돌리려면 이 상수만 RISE_ST 로 바꾸면 된다.
#
# 2026-08-11 보탬: 이 값은 그대로 둔다. 변화량으로 환산하면 3.5반음이라
# 이미 지각 문턱(1.5~2반음)의 두 배이고, 5세션 실측에서도 발화의 17.1%에
# 붙어 "?" 로서 과하지 않다. 다만 위 문단의 "RISE_ST(=1.5)는 건드리지
# 않았으므로 화살표는 종전 그대로"라는 서술은 더 이상 사실이 아니다 —
# 화살표 문턱이 같은 날 1.5 st/s → 4.29 st/s(=1.5반음)로 올라갔다.
# 두 계열이 서로 다른 상수를 쓴다는 구조는 그대로다.
QUESTION_ST = 10.0

# 음량(° °, 대문자) 판정 — 화자 기준선 대비 상대값.
#
# 배경 (2026-08-08 CA 데이터 세션, Tim Greer 지적): "전사본의 거의 모든 턴이
# 조용(°)하게 표시된다. 왜 'of' 같은 낱말이 조용하다고 표시되나."
#
# 원인은 두 가지였고 **둘 다 지표 쪽 문제**였다.
#   ① 낱말 음량을 프레임 **중앙값**으로 재고 있었다. Praat 강도 곡선은
#      약 53ms 창으로 평활되므로, 침묵 직후 첫 낱말에는 바닥에서 올라오는
#      상승 램프 프레임이 절반 가까이 섞인다. 중앙값이 그 램프에 끌려
#      내려간다 → 발화 첫 낱말이 구조적으로 조용해진다.
#   ② 기준선이 발화 스팬 전체의 프레임 중앙값(speaker_db_median)이었다.
#      낱말 사이 저강도 프레임이 섞여 낱말끼리의 비교가 성립하지 않았다.
#
# 실측 (2026-08-11, GPT 코퍼스 5세션 · 낱말 20,432 · 화자 11명):
#   현행 지표(프레임 중앙값 − 발화스팬 기준선) 분위
#     p1=-25.1  p5=-15.1  p10=-8.7  p25=-2.6  p50=+0.7  p95=+5.5  p97=+6.1
#   → QUIET_DB=-6.0 은 이 분포에서 상위 12% 지점이라, 낱말의 **13.7%**가
#     quiet 이었다. 발화 첫 낱말 편향 3.04배(기준선 11.2% → quiet 중 34.1%),
#     quiet 상위 낱말은 전부 기능어(and · the · of · to · a · i · it's).
#
#   새 지표(낱말 프레임 p75 − 낱말 p75 의 화자 중앙값) 분위
#     p1=-20.2  p2=-13.5  p3=-10.3  p5=-7.7  p10=-4.9  p25=-1.9  p50=0.0
#     p75=+1.3  p90=+2.8  p95=+3.8  p97=+4.8  p98=+5.6  p99=+6.9
#
# p75 로 바꾼 이유: 낱말의 상위 4분위 프레임은 모음 핵(가장 소리가 실린
# 지점)이고, 램프·마찰음 프레임은 아래쪽에 몰린다. 즉 p75 는 "이 낱말이
# 얼마나 크게 났나"를 재고 중앙값은 "이 낱말 구간이 평균적으로 얼마나
# 시끄러웠나"를 잰다. 산포도 절반으로 줄었다(p5: -15.1 → -7.7).
#
# 값 선택:
#   QUIET_DB = -7.5  ← 새 지표 p5(-7.67)를 0.5dB 단위로 반올림. "그 화자
#       기준 하위 5%"라는 한 문장으로 설명되는 값이다. 여기에 연속 2낱말
#       규칙(analysis.detect_prosody)이 겹쳐 실제 표기는 더 줄어든다.
#
#       문턱별 실측 (같은 5세션, 고립 제외 = 실제 ° 표기율):
#         문턱   낱말단위   고립제외   run수
#         -5.0    9.73%     1.86%     171
#         -6.0    7.42%     1.02%      95
#         -7.0    5.79%     0.59%      56
#         -7.5    5.17%     0.51%      48   ← 채택
#         -8.0    4.64%     0.38%      36
#       0.51% 는 세션당 ° 구간 9곳 꼴이다. 종전 13.7%(세션당 550곳 이상)와
#       비교할 값이 아니다. 표기를 늘리려면 -6.0(약 1%)이 다음 후보이고,
#       고립 제외 규칙과 문턱은 서로 곱해지므로 반드시 함께 봐야 한다.
#   LOUD_DB  = +5.0  ← 새 지표 p97(+4.82) 자리. 상위 약 2.5%.
#       종전 +6.0 은 현행 지표의 p97(+6.1)이었으므로 **같은 분위를 새
#       지표에서 다시 찾은 것**이고, 대문자 표기량은 그대로 유지된다.
#
# 되돌리려면 이 두 상수와 word_volume() 의 지표(p75→중앙값)를 함께 되돌려야
# 한다. 문턱만 되돌리면 분포가 달라 의미가 없다.
QUIET_DB = -7.5      # 낱말 p75 가 화자 기준선보다 이만큼 낮으면 작은 소리
LOUD_DB = +5.0       # 이만큼 높으면 큰 소리
WORD_MIN_FRAMES = 3  # 강도 프레임이 이보다 적으면 음량 판정을 하지 않는다

# 강조(stress) 판정 — 화자 기준선 대비 상대값.
#
# 지시받은 초기값은 +6dB / +4반음이었는데, 실측(S-02, 낱말 3325개, 화자
# 3명)에서 낱말의 8.7%·발화의 73%에 밑줄이 붙었다. 강조가 아니라 배경이
# 되어 버린다. 화자별 분포를 재 보니:
#
#   낱말 평균 강도 편차   p90 ≈ +4.5dB   p95 ≈ +5.0dB   p99 ≈ +6.7dB
#   낱말 평균 F0 편차*    p90 ≈ +4.2st   p95 ≈ +5.2st   p99 ≈ +6.5st
#   (*옥타브 오류 제거 후. 제거 전에는 p95 가 화자에 따라 5.8~17.7st 로
#     널뛰었다 — 아래 OCTAVE_LIM_ST 주석 참조)
#
# 즉 +4반음은 상위 10%여서 '강조'라 부를 수 없다. 세 화자 모두 p99 가
# 약 +7 로 모여 있어 양쪽 다 +7 로 맞췄다 — "그 화자 기준 상위 1%"라는
# 한 문장으로 설명되는 값이다. 결과: 낱말의 1.0%, 발화의 13%.
STRESS_DB = +7.0     # 낱말 평균 강도가 화자 중앙값보다 이만큼 높으면 강조
STRESS_ST = +7.0     # 낱말 평균 F0 가 화자 중앙값보다 이만큼(반음) 높으면 강조

# 옥타브 오류 제거 폭(반음). 화자 중앙값에서 이보다 멀리 떨어진 F0 프레임은
# 강조 판정에서 버린다.
#
# 필요한 이유: Praat 의 피치 추적은 PITCH_FLOOR~PITCH_CEIL 이 넓으면
# 기본 주파수를 2배(또는 1/2배)로 잘못 잡는 일이 있다. 실측에서 화자 A·C 의
# 낱말 F0 편차 상위값이 +17~+22반음이었는데, 12반음이 정확히 한 옥타브다 —
# 사람이 한 낱말에서 한 옥타브 반을 올려 말하지는 않는다. 이 프레임들이
# 강조 검출의 대부분(288개 중 267개)을 차지하고 있었다.
# ±9반음으로 자르면 세 화자의 분포가 서로 겹칠 만큼 정돈된다
# (p95: 5.5 / 5.1 / 5.2). 실제 강조는 +7반음 안쪽이므로 잘리지 않는다.
OCTAVE_LIM_ST = 9.0
STRESS_MIN_VOICED = 3   # 이보다 유성 프레임이 적으면 F0 로 판정하지 않는다


class ProsodyTrack:
    """한 오디오의 F0·강도 궤적. 시각으로 조회한다.

    조회는 이진탐색으로 구간을 먼저 자른다. 시간축이 오름차순이므로
    `bisect_left(start)`~`bisect_right(end)`가 `start <= t <= end`와 정확히
    같은 구간이고, 전체 스캔이 없어진다.

    이게 중요한 이유: 32분 파일이면 프레임이 15만 개인데 `word_volume()`이
    **단어마다** `db_in`을 부른다. 단어 3800개면 전체 스캔으로 약 5억 회
    반복이었다. 이제 O(log n + 구간길이)다.
    """

    def __init__(self, pitch_t, pitch_f, int_t, int_db, audio_path=None):
        self.pitch_t, self.pitch_f = pitch_t, pitch_f
        self.int_t, self.int_db = int_t, int_db
        # 중대역 강도·파형은 **지연 계산**이다 — audio_path 만 들고 있다가
        # 처음 요청받을 때 한 번 만든다. 이유는 mid_band_ratio()·cog() 주석 참조.
        self._audio_path = audio_path
        self._mid_db = None
        self._snd = None
        self._snd_tried = False

    # ── 조회 ──
    def f0_in(self, start: float, end: float) -> List[Tuple[float, float]]:
        lo = bisect.bisect_left(self.pitch_t, start)
        hi = bisect.bisect_right(self.pitch_t, end)
        return [(t, f) for t, f in zip(self.pitch_t[lo:hi],
                                       self.pitch_f[lo:hi]) if f > 0]

    def db_in(self, start: float, end: float) -> List[float]:
        lo = bisect.bisect_left(self.int_t, start)
        hi = bisect.bisect_right(self.int_t, end)
        return [d for d in self.int_db[lo:hi] if d > 0]

    # ── 중대역(500~4000Hz) 강도 — 소음 판정 전용 ──
    def mid_band_ratio(self, start: float, end: float) -> Optional[float]:
        """구간 에너지 중 **500~4000Hz 대역이 차지하는 비중**(dB, 0이 전부).

        detect_noise() 가 "무성 **광대역**"을 확인하는 데 쓴다. 전대역 강도만
        보면 광대역 소음(박수·의자·문)과, 낱말 끝에서 새어 나온 치찰음
        (/s/ /z/ /ks/) 과, 마이크에 닿은 초저역 충격음이 구분되지 않는다.
        셋 다 '크고 무성'이기 때문이다. 대역 비중은 그 셋을 갈라 놓는다.

        실측 (2026-08-12, GPT 5세션 · 소음 후보 21곳, FFT 대조 확인):
          광대역(박수·의자류 후보)  −2.1 ~ −4.6dB   (중대역이 35~62%)
          낱말 끝 치찰음 누출        −13.7 ~ −24.2dB (에너지 96~98%가 4kHz 위)
          초저역 충격·DC 아티팩트    −18.3 ~ −36.2dB (에너지 99%가 500Hz 아래)
        두 무리 사이가 3dB 이상 비어 있어 −6dB(=중대역 25%)로 끊었다.

        **지연 계산**인 이유: Praat 의 Hann 대역통과는 파일 전체를 FFT 하므로
        33분 파일에서 18~25초가 든다(전대역 강도 추출은 2.5초). 소음 판정을
        하지 않는 호출부(intonation_all·volume_check·intonation_check·
        화자검증/cutoff_acoustic)까지 그 값을 물리면 안 된다."""
        if self._mid_db is None:
            self._mid_db = self._build_mid()
        if not self._mid_db:
            return None
        lo = bisect.bisect_left(self.int_t, start)
        hi = bisect.bisect_right(self.int_t, end)
        pairs = [(m, f) for m, f in zip(self._mid_db[lo:hi], self.int_db[lo:hi])
                 if f > 0 and m > 0]
        if len(pairs) < 3:
            return None
        import statistics
        return (statistics.median(m for m, _ in pairs)
                - statistics.median(f for _, f in pairs))

    # ── 구간 스펙트럼 무게중심 — 들숨 판별 전용 ──
    def cog(self, start: float, end: float) -> Optional[float]:
        """구간 스펙트럼의 **무게중심(centre of gravity, Hz)**. 실패하면 None.

        들숨은 성문 아래에서 나는 난기류 소음이라 에너지가 중저역에 퍼진다.
        같은 대역·같은 무성 조건을 통과하면서도 무게중심이 높은 것은 들숨이
        아니라 치찰음 누출(/s/ /ㅅ/)이나 딸깍 소리다 — 값 근거는
        detect_breaths() 주석의 판정 정답지 성적표 참조.

        mid_band_ratio() 와 달리 **구간마다 다른 값**이라 트랙으로 미리
        만들어 둘 수 없다. 파형(Sound)만 한 번 올려 두고 구간을 잘라 쓴다.
        파형 적재도 지연이라, 들숨 후보가 없으면 비용이 0이다."""
        snd = self._sound()
        if snd is None:
            return None
        try:
            part = snd.extract_part(from_time=start, to_time=end,
                                    preserve_times=False)
            if part.get_total_duration() <= 0.01:
                return None
            return part.to_spectrum().get_centre_of_gravity(power=2)
        except Exception:
            return None

    def _sound(self):
        if not self._snd_tried:
            self._snd_tried = True
            try:
                import parselmouth
                self._snd = parselmouth.Sound(self._audio_path) \
                    if self._audio_path else None
            except Exception:
                self._snd = None
        return self._snd

    def _build_mid(self) -> list:
        """중대역 강도를 전대역과 **같은 시간축**으로 만든다. 실패하면 []."""
        if not self._audio_path:
            return []
        try:
            import parselmouth
            from parselmouth.praat import call
            snd = parselmouth.Sound(self._audio_path)
            band = call(snd, "Filter (pass Hann band)...",
                        MID_BAND_LO, MID_BAND_HI, MID_BAND_SMOOTH)
            it = band.to_intensity(minimum_pitch=PITCH_FLOOR)
            return [it.get_value(time=t) or 0.0 for t in self.int_t]
        except Exception:
            return []


_WARNED = False


_WARNED_FMT = [False]


def _to_wav(src: str) -> Optional[str]:
    """운율 분석용으로 모노 16kHz WAV 를 임시 추출한다. 실패하면 None."""
    try:
        import av
        import numpy as np
        import tempfile
        import wave
        c = av.open(src)
        st = c.streams.audio[0]
        res = av.AudioResampler(format="s16", layout="mono", rate=16000)
        buf = []
        for fr in c.decode(st):
            for rf in res.resample(fr):
                buf.append(rf.to_ndarray().reshape(-1))
        c.close()
        if not buf:
            return None
        data = np.concatenate(buf).astype("<i2")
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes(data.tobytes())
        return path
    except Exception:
        return None


def extract(audio_path: str) -> Optional[ProsodyTrack]:
    """오디오에서 F0·강도 궤적을 뽑는다. 실패하면 None (분석은 계속 진행).

    parselmouth 가 없으면 억양(`↗ ↘`)·음량(`° °`)·호흡 마커가 **통째로
    빠진다.** 예전에는 조용히 넘어가서, 같은 연구의 전사본인데 표기 체계가
    갈리는 일이 생겼다. 이제 한 번 경고한다.
    """
    global _WARNED
    try:
        import parselmouth
    except ImportError:
        if not _WARNED:
            _WARNED = True
            print("    [경고] parselmouth 가 없어 운율 분석을 건너뜁니다 — "
                  "억양(^˅)·음량(° °)·호흡 마커가 생성되지 않습니다.\n"
                  "           설치:  pip install praat-parselmouth",
                  file=sys.stderr)
        return None
    # parselmouth 는 WAV/AIFF 계열만 읽는다. 우리 코퍼스는 캠코더 MTS 와
    # mp4 가 많은데, 그런 파일에서 Sound() 가 던지는 예외를 아래 except 가
    # 통째로 삼켜 **운율이 조용히 사라졌다**(2026-08-13, 76세션 배치 중 발견).
    # 경고는 parselmouth 미설치만 잡고 있어서 이 경로는 아무 표시도 없었다.
    # 컨테이너를 가리지 않도록 PyAV 로 모노 16kHz WAV 를 임시 추출해 넘긴다.
    tmp_wav = None
    if os.path.splitext(audio_path)[1].lower() not in (".wav", ".aiff", ".aifc"):
        tmp_wav = _to_wav(audio_path)
        if tmp_wav is None:
            if not _WARNED_FMT[0]:
                _WARNED_FMT[0] = True
                print(f"    [경고] 운율용 오디오 변환 실패 — 억양·음량·호흡이 "
                      f"생성되지 않습니다: {os.path.basename(audio_path)}",
                      file=sys.stderr)
            return None
        audio_path = tmp_wav
    try:
        snd = parselmouth.Sound(audio_path)
        pitch = snd.to_pitch(pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEIL)
        inten = snd.to_intensity(minimum_pitch=PITCH_FLOOR)
        return ProsodyTrack(
            list(pitch.xs()),
            [pitch.get_value_at_time(t) or 0.0 for t in pitch.xs()],
            list(inten.xs()),
            [inten.get_value(time=t) or 0.0 for t in inten.xs()],
            audio_path,
        )
    except Exception as e:
        if not _WARNED_FMT[0]:
            _WARNED_FMT[0] = True
            print(f"    [경고] 운율 분석 실패 ({type(e).__name__}) — "
                  f"억양·음량·호흡이 생성되지 않습니다", file=sys.stderr)
        return None
    finally:
        if tmp_wav:
            try:
                os.unlink(tmp_wav)
            except OSError:
                pass


# ── 억양 ────────────────────────────────────────────────────────────────
def _semitone_slope(pts: List[Tuple[float, float]]) -> Optional[float]:
    """말미 F0 기울기를 반음/초로. 반음은 로그 척도라 성별 차를 흡수한다."""
    if len(pts) < 3:
        return None
    import math
    t0, f0 = pts[0]
    span = pts[-1][0] - t0
    if span <= 0 or f0 <= 0:
        return None
    st = [12 * math.log2(f / f0) for _, f in pts if f > 0]
    if len(st) < 3:
        return None
    # 단순 최소제곱 기울기
    ts = [t - t0 for t, f in pts if f > 0]
    n = len(st)
    mt, ms = sum(ts) / n, sum(st) / n
    den = sum((t - mt) ** 2 for t in ts)
    if den == 0:
        return None
    return sum((t - mt) * (s - ms) for t, s in zip(ts, st)) / den


def utterance_slope(track: ProsodyTrack, start: float, end: float,
                    base_f0: Optional[float] = None) -> Optional[float]:
    """발화 말미(TAIL_SEC)의 F0 기울기(반음/초). 판정 불가면 None.

    등급 판정과 분리해 둔 이유: 화살표(↗↘)와 말미 구두점(? , .)이 서로 다른
    문턱을 쓸 수 있어야 한다. 하나의 함수가 등급만 돌려주면, 구두점 문턱을
    올리는 순간 화살표까지 따라 사라진다 — 교수님이 "화살표는 화살표대로
    쓰라"고 하신 것과 어긋난다. 기울기를 한 번만 재고 두 판정이 나눠 쓴다.

    base_f0 (화자 F0 중앙값, Hz)를 주면 그 값에서 ±OCTAVE_LIM_ST 반음보다
    멀리 떨어진 말미 프레임을 **버리고** 기울기를 낸다.

    왜 필요한가 (2026-08-11, 지각 문턱 재보정의 연장): 문턱을 지각 수준
    (말미 1.5반음)으로 올리고도 화살표가 발화의 53%에 붙었다. 입력 분포를
    보면 이유가 나온다 — 말미 기울기 |값|의 **중앙값이 11.8 반음/초, 즉
    말미 0.35초 동안 4.1반음**이다. 사람이 말끝에 4반음을 움직이는 일이
    발화의 절반일 리 없다. Praat 피치 추적이 기본 주파수를 2배(또는 1/2배)로
    잡는 옥타브 점프가 말미에 그대로 들어온 것이고, 한 옥타브는 정확히
    12반음이라 기울기 한두 프레임이 곧바로 수십 반음/초가 된다.
    강조 판정이 이미 같은 이유로 같은 폭(±9반음)을 쓰고 있다 —
    OCTAVE_LIM_ST 주석 참조. 문턱이 아니라 **입력**을 고치는 쪽이다.

    실측 효과 (GPT 5세션 · 발화 2,304): 필터를 켜면 말미 |기울기| 분위가
      p25 5.0→4.5 · p50 11.8→10.0 · p75 26.2→21.1 · p90 49.3→34.9
    로 내려가고, 부착률은 화살표 53.4%→**47.6%** · "?" 17.1%→**14.2%** ·
    "." 29.8%→26.7% 가 된다. ","(6.6%→6.8%)만 조금 늘었는데, 옥타브 점프로
    "?" 로 과판정되던 발화가 제 자리인 "약한 상승"으로 내려온 것이다.

    기본값 None 은 **필터 없음**이다. 화자 중앙값을 계산하지 않는 외부
    스크립트(intonation_all.py·intonation_check.py)가 종전 동작 그대로
    돌아가야 하므로 옵션으로 뒀다."""
    if track is None:
        return None
    tail = track.f0_in(max(start, end - TAIL_SEC), end)
    if base_f0:
        lo = base_f0 * 2 ** (-OCTAVE_LIM_ST / 12)
        hi = base_f0 * 2 ** (OCTAVE_LIM_ST / 12)
        tail = [(t, f) for t, f in tail if lo <= f <= hi]
    return _semitone_slope(tail)


def intonation_arrow(slope: Optional[float]) -> Optional[str]:
    """화살표 표기용 2등급. 'rise'(↗) | 'fall'(↘) | None(평탄).

    RISE_ST / FALL_ST 를 쓴다 — 구두점(intonation_grade)과 상수를 공유하되
    등급이 다르다. 2026-08-11 에 두 상수가 지각 문턱(말미 1.5반음)으로
    올라가면서 화살표 부착률이 발화의 62.3% → 53.4% 로 줄었고, 같은 날
    말미의 옥타브 오류 프레임을 버리면서(utterance_slope 의 base_f0)
    47.6% 가 됐다. 문턱과 입력을 각각 한 번씩 고친 결과다."""
    if slope is None:
        return None
    if slope >= RISE_ST:
        return "rise"
    if slope <= FALL_ST:
        return "fall"
    return None


def utterance_intonation(track: ProsodyTrack, start: float, end: float,
                         base_f0: Optional[float] = None) -> Optional[str]:
    """발화 말미의 억양 방향.

    'rise' | 'weak_rise' | 'fall' | None(평탄·판정불가)

    3등급으로 나눈 이유 (근거: 2026-08-06 교수님 미팅 지시). 기존 2등급
    (rise/fall)은 화살표 ↗↘ 를 붙이는 데는 충분했지만, CA 전사본이 관례로
    쓰는 말미 구두점 `. , ?` 을 만들 수 없었다. 세 등급은 각각
      rise      (>= QUESTION_ST, 말미 3.5반음) → "?"  질문식 강한 상승
      weak_rise (RISE_ST ~ QUESTION_ST,
                 말미 1.5~3.5반음)            → ","  "아직 안 끝났다" 계속 신호
      fall      (<= FALL_ST, 말미 -1.5반음)   → "."  종결식 하강
    에 대응한다. 세 등급의 하한은 전부 **지각 문턱**에서 왔다 —
    RISE_ST/FALL_ST 주석의 2026-08-11 재보정 참조.

    ⚠️ 이 구두점은 **억양 윤곽의 표기**이지 **질문 여부의 판정이 아니다.**
    개발기록 2.11 절 참조 — 한국어의 정중한 의문형("~하시겠습니까")은
    끝음이 오히려 **내려간다**. 그러므로 여기서 "." 이 붙은 발화가 평서문
    이라는 보장도, "?" 가 붙은 발화가 질문이라는 보장도 없다. Jefferson
    표기에서 `.` `,` `?` 가 원래 문법 부호가 아니라 억양 기호인 것과 같다.

    base_f0 는 utterance_slope() 로 그대로 넘어간다(옥타브 오류 제거).
    주지 않으면 필터 없이 종전대로 동작한다."""
    return intonation_grade(utterance_slope(track, start, end, base_f0))


def intonation_grade(slope: Optional[float]) -> Optional[str]:
    """구두점 표기용 3등급. 'rise' | 'weak_rise' | 'fall' | None."""
    if slope is None:
        return None
    if slope >= QUESTION_ST:
        return "rise"
    if slope >= RISE_ST * WEAK_RISE_FRAC:
        return "weak_rise"
    if slope <= FALL_ST:
        return "fall"
    return None


# ── 음량 ────────────────────────────────────────────────────────────────
def speaker_db_median(track: ProsodyTrack,
                      spans: List[Tuple[float, float]]) -> Optional[float]:
    import statistics
    vals = [d for s, e in spans for d in track.db_in(s, e)]
    return statistics.median(vals) if vals else None


def word_db_p75(track: ProsodyTrack, start: float,
                end: float) -> Optional[float]:
    """낱말 구간 강도의 **p75 분위값**(dB). 프레임이 3개 미만이면 None.

    중앙값이 아니라 p75 인 이유는 QUIET_DB 주석 참조 — 낱말 앞뒤의 상승·하강
    램프 프레임과 무성 자음 프레임이 아래쪽 분위를 채우므로, 중앙값은 "이
    낱말이 얼마나 크게 났나"가 아니라 "이 구간이 평균적으로 얼마나 시끄러웠나"
    를 재게 된다. 상위 4분위는 모음 핵에 해당해 램프의 영향을 받지 않는다.

    프레임 3개(≈40ms) 미만을 버리는 이유: 강도 프레임 간격이 약 13ms
    (Praat 는 minimum_pitch 60Hz 에서 0.8/60 초 간격)이라, 2개면 분위값이
    사실상 두 점 사이 보간이라 값이 튄다."""
    import statistics
    vals = track.db_in(start, end)
    if len(vals) < WORD_MIN_FRAMES:
        return None
    return statistics.quantiles(vals, n=4, method="inclusive")[2]


def speaker_word_p75_median(track: ProsodyTrack,
                            word_spans: List[Tuple[float, float]]
                            ) -> Optional[float]:
    """**낱말 p75** 의 화자 중앙값. 음량(° °, 대문자) 판정의 기준선이다.

    speaker_db_median() 과 다른 점: 저쪽은 발화 스팬의 프레임을 전부 한 통에
    넣은 중앙값이라 낱말 사이의 저강도 프레임(자음·짧은 간격·꼬리)이 섞인다.
    비교 대상(낱말 p75)과 단위가 달라 '주변보다 조용한가'를 물을 수 없었다.
    양쪽을 같은 낱말 단위로 맞추면 QUIET_DB 가 곧 "이 화자의 보통 낱말보다
    몇 dB 낮은가"가 된다 (speaker_word_db_median 과 같은 논리이고, 저쪽은
    강조 판정용 평균 기준선이라 별도로 남겨 둔다)."""
    import statistics
    vals = []
    for s, e in word_spans:
        p = word_db_p75(track, s, e)
        if p is not None:
            vals.append(p)
    return statistics.median(vals) if vals else None


def word_volume(track: ProsodyTrack, start: float, end: float,
                base_db: float) -> Optional[str]:
    """'quiet' | 'loud' | None. 화자 낱말 기준선 대비 상대 판정.

    base_db 는 speaker_word_p75_median() 값이어야 한다 — 지표가 낱말 p75 라
    기준선도 같은 단위여야 문턱(QUIET_DB/LOUD_DB)의 의미가 유지된다.
    시그니처는 종전과 같으므로 호출부는 기준선 계산만 바꾸면 된다."""
    if track is None or base_db is None:
        return None
    p75 = word_db_p75(track, start, end)
    if p75 is None:
        return None
    d = p75 - base_db
    if d <= QUIET_DB:
        return "quiet"
    if d >= LOUD_DB:
        return "loud"
    return None


# ── 강조 ────────────────────────────────────────────────────────────────
# (근거: 2026-08-06 교수님 미팅 지시 — 강조 낱말을 밑줄로 표시)
def speaker_f0_median(track: ProsodyTrack,
                      spans: List[Tuple[float, float]]) -> Optional[float]:
    """화자의 F0 중앙값(Hz). speaker_db_median 과 같은 방식이다.

    강조 판정의 기준선이다. 절대 Hz 로는 판정할 수 없으므로(성별·개인차)
    이 값 대비 반음 차이로 본다."""
    import statistics
    vals = [f for s, e in spans for _, f in track.f0_in(s, e)]
    return statistics.median(vals) if vals else None


def speaker_word_db_median(track: ProsodyTrack,
                           word_spans: List[Tuple[float, float]]
                           ) -> Optional[float]:
    """**낱말 평균 강도**의 화자 중앙값. 강조 판정의 기준선이다.

    speaker_db_median() 과 다른 점: 저쪽은 프레임을 전부 한 통에 넣은
    중앙값이라 낱말 사이의 낮은 프레임까지 섞인다. 강조는 '이 낱말이 이
    화자의 보통 낱말보다 센가'를 묻는 것이므로 기준선도 낱말 단위여야
    비교가 성립한다 (실측 차이는 화자에 따라 0.0~1.0dB — 문턱이 7dB 인
    판정에서 작지 않다)."""
    import statistics
    vals = []
    for s, e in word_spans:
        d = track.db_in(s, e)
        if len(d) >= 2:
            vals.append(statistics.mean(d))
    return statistics.median(vals) if vals else None


def word_stress(track: ProsodyTrack, start: float, end: float,
                base_db: Optional[float],
                base_f0: Optional[float]) -> Optional[str]:
    """낱말이 강조되었는지. 'db' | 'f0' | None.

    강도(dB)나 음높이(F0) 중 **하나라도** 화자 기준선을 크게 넘으면 강조로
    본다. CA 에서 강세는 크게 말하거나 높게 말하거나 둘 다인데, 둘을 AND 로
    묶으면 거의 잡히지 않는다.

    F0 쪽은 옥타브 오류 프레임을 먼저 버린다(OCTAVE_LIM_ST 주석 참조).
    이 걸러내기가 없으면 강조 검출이 사실상 피치 추적 실패 검출기가 된다.

    반환값을 bool 이 아니라 근거 문자열로 주는 이유: 마커 info 에 남겨
    나중에 '음량형 강조'와 '음높이형 강조'를 갈라 볼 수 있게 하기 위함이다.
    volume_loud(+6dB)와 겹칠 수 있는데, 겹침 자체는 문제가 아니다 —
    대문자는 '크게', 밑줄은 '강조'로 CA 에서 원래 다른 층위다."""
    if track is None:
        return None
    import math
    import statistics
    if base_db is not None:
        vals = track.db_in(start, end)
        if len(vals) >= 2 and statistics.mean(vals) - base_db >= STRESS_DB:
            return "db"
    if base_f0 is not None and base_f0 > 0:
        lo = base_f0 * 2 ** (-OCTAVE_LIM_ST / 12)
        hi = base_f0 * 2 ** (OCTAVE_LIM_ST / 12)
        f0 = [f for _, f in track.f0_in(start, end) if lo <= f <= hi]
        if len(f0) >= STRESS_MIN_VOICED:
            st = 12 * math.log2(statistics.mean(f0) / base_f0)
            if st >= STRESS_ST:
                return "f0"
    return None


# ── 호흡 ────────────────────────────────────────────────────────────────
def detect_breaths(track: ProsodyTrack, gaps: List[Tuple[float, float]],
                   base_db: float) -> List[Tuple[float, float, str]]:
    """단어 사이 구간 중 '소리는 있으나 음정이 없는' 곳을 호흡으로 본다.

    무음과의 구별: 강도가 바닥보다 뚜렷이 높은데 F0가 잡히지 않으면
    무성 마찰음(들숨·날숨)일 가능성이 크다. 보수적으로만 표시한다.

    강도 대역 재보정 (2026-08-11, 지각 문턱 일괄 재보정):
      사람 전사자는 세션당 20~39곳에 `.hh` 를 적는다(output.py 주석).
      종전 대역(기준선 −20 ~ −5dB)에서는 세션당 평균 **95.4곳**이었다 —
      아래로는 방 소음·에어컨까지, 위로는 말꼬리의 저강도 프레임까지
      들숨으로 셌다는 뜻이다.
      실측 (GPT 5세션, 무성·저강도 후보 956구간의 기준선 대비 dB):
        p10=-27.5  p25=-23.0  p50=-18.0  p75=-11.9  p90=-6.3  p95=-2.4
      대역별 세션당 개수 —
        −20~−5(종전) 95.4 · −18~−8 67.4 · −15~−10 36.8 · −14~−9 33.8 ·
        **−12~−8 20.2** · −10~−8 8.2
      −12~−8 을 골랐다. 다섯 세션 모두 39곳을 넘지 않는 유일한 대역이고
      (39·35·9·16·2), 지시대로 '적게 잡는 쪽'이다. 들숨은 들리기는 해도
      말소리보다 확실히 작다는 상식과도 맞는다 — 기준선보다 8dB 이상
      낮아야 하고, 12dB 넘게 낮으면 그건 이미 무음이다.

    판별 게이트 신설 (2026-08-12) — **개수가 맞았다고 내용이 맞은 것은
    아니었다.** 위 재보정은 세션당 개수를 사람 범위(20~39)에 맞춘 것이지
    "그 자리가 정말 들숨인가"를 확인한 것이 아니다. 연구자가
    참여자A(S-01) 세션의 후보 46건을 **직접 듣고** 판정했다
    (정답지: `(내부 검증 csv, 비공개)` — 맞음 13 · 아님 32 · 애매 1).
    개수는 사람 범위 안이었지만 **정밀도는 28.3%** 였다.

    오탐의 정체는 정답지 메모에 그대로 적혀 있다 — "딸깍 소리"(2건) ·
    "'응대::' 같은 늘임" · "'감?' 하고 올라가는 억양 꼬리" · "'판'에 숨 소리" ·
    "'엮으면'에서 부스럭 소리" · "'yes' 끝에 숨 섞임" · "연구원이 문 열고
    나가는 소리". 즉 (ㄱ) 낱말에 붙어 새어 나온 마찰음·숨, (ㄴ) 클릭류 충격음,
    (ㄷ) 평탄한 부스럭 잡음 셋이다. 게이트 셋을 그 셋에 각각 맞췄다.

      ① `BREATH_COG_MAX` 3000Hz — 스펙트럼 무게중심 상한. 들숨은 난기류
         소음이라 중저역에 퍼지고, 치찰음 누출·딸깍은 무게중심이 높다
         (오탐 실측 4563 · 4747 · 4798 · 4966 · 4975 · 6146Hz).
      ② `BREATH_ENV_MIN`~`MAX` 6~14dB — 포락선(구간 피크dB − 중앙dB).
         하한 미만은 시작도 끝도 없는 **평탄한 잡음**(부스럭·에어컨),
         상한 초과는 한 프레임만 튀는 **클릭**이거나 늘임 꼬리다.
      ③ `BREATH_MIN` 0.15 → 0.20초. 사람이 들숨으로 알아듣는 최소 길이.

    정답지 성적표 (같은 46건):

      | 규칙 | 유지 맞음 | 제거 아님 | 남은 총계 | 정밀도 |
      |---|---|---|---|---|
      | 현행(게이트 없음) | 13/13 | 0/32 | 46 | 28.3% |
      | 규칙1 — ①만 | 12/13 | 6/32 | 39 | 30.8% |
      | **규칙2 — ①②③(채택)** | **11/13** | **21/32** | **23** | **47.8%** |
      | 규칙3 — 규칙2 + 길이 0.24초 | 8/13 | 25/32 | 15 | 53.3% |

    규칙2 를 골랐다. 규칙3 은 정밀도를 5.5pp 더 올리는 대신 **맞음을 3건
    더 잃는다**(11→8). 이 마커의 용도는 사람 전사자가 검토할 후보를 내는
    것이므로, 재현율을 그만큼 내주고 얻는 정밀도가 아니다. 규칙2 가 놓치는
    맞음 2건은 포락선 15.66dB(상한 초과)과 무게중심 5329Hz(상한 초과)라
    둘 다 게이트의 정의상 걸린 것이고, 문턱을 그 둘에 맞춰 늘리면 오탐이
    함께 돌아온다.

    **기각한 특징**: 직전 낱말과의 간격(`prev_gap`)은 46건 **전부 0.000초**라
    판별력이 없었다 — 이 코퍼스는 낱말 스팬이 뒤따르는 침묵을 삼키고 있어
    낱말 사이 간격이 원리적으로 0 이다(2.35절 ①). 직전 낱말의 치찰음 종성
    여부도 갈리지 않았다.

    ⚠️ **dB 값의 이식성**: 위 포락선 대역은 참여자A 세션의 wav 변환본에서 잰
    값이고, 같은 구간을 다시 재면 원본 대비 ±0.5dB 편차가 있었다(mp3 →
    wav 변환·디코더 차이). 6~14dB 은 이 정답지 기준이며, 녹음 조건이 다른
    코퍼스에 옮길 때는 재검이 필요하다.

    ⚠️ **GPT(TTS) 숨소리**: 정답지의 후보 대부분이 GPT 발화 구간에 있고
    연구자는 "GPT가 들숨이 어디 있어"라며 아님으로 판정했다. 다만 8/8 데이터
    세션에서 그 소리는 'AI 를 인간처럼 느끼게 하는 장치'로 논의된 현상이다.
    전사 대상에 넣을지는 연구 프레임의 결정이라 코드로 정하지 않았다
    (보고서 결정 요청 D4)."""
    if track is None or base_db is None:
        return []
    import statistics
    out = []
    for s, e in gaps:
        dur = e - s
        if not (BREATH_MIN <= dur <= BREATH_MAX):
            continue
        db = track.db_in(s, e)
        f0 = track.f0_in(s, e)
        if len(db) < 2:
            continue
        med = statistics.median(db)
        # 무음보다는 크고(기준선 −BREATH_DB_FLOOR 이상) 말소리보다는 작으며
        # (기준선 −BREATH_DB_CEIL 이하), 음정이 거의 잡히지 않을 것
        if not (med >= base_db - BREATH_DB_FLOOR
                and med <= base_db - BREATH_DB_CEIL and len(f0) <= 1):
            continue
        # ② 포락선 — 시작과 끝이 있는 소리인가
        env = max(db) - med
        if not (BREATH_ENV_MIN <= env <= BREATH_ENV_MAX):
            continue
        # ① 무게중심 — 난기류 소음인가, 치찰음·딸깍인가
        #
        # 갭 가장자리를 **자르지 않는다**. detect_noise() 는 0.08초를 트림해
        # 낱말에서 새어 나온 조각을 버리는데, 들숨은 애초에 낱말에 바싹 붙어
        # 나는 소리라 트림하면 신호 자체가 사라진다. 위 성적표도 트림 없이
        # 잰 값이므로 재현을 위해서도 여기서는 자르지 않는다.
        cog = track.cog(s, e)
        if cog is None or cog > BREATH_COG_MAX:
            continue
        out.append((s, e, f"dur={dur:.2f}s db={med - base_db:+.1f} "
                          f"cog={cog:.0f} env={env:.1f}"))
    return out


# ── 배경 소음 ───────────────────────────────────────────────────────────
def detect_noise(track: ProsodyTrack, gaps: List[Tuple[float, float]],
                 base_db: float,
                 exclude: Optional[List[Tuple[float, float]]] = None
                 ) -> List[Tuple[float, float, str]]:
    """어느 화자의 낱말에도 속하지 않는 갭 중 **크고 무성이며 광대역**인 곳.

    Jefferson 관례로 전사자가 ((문 닫는 소리)) 처럼 이중 괄호에 적는 것이다.
    박수·의자 끄는 소리·문소리는 성대 진동이 없고(F0 안 잡힘) 에너지가
    넓은 대역에 퍼져 있으며, 말소리에 맞먹게 크다.

    detect_breaths() 와 골격이 같다 — 같은 낱말 사이 갭을 훑고, 강도
    중앙값과 유성 프레임 수를 본다. 다른 것은 **대역의 방향**이다.
    들숨은 기준선보다 8~12dB 낮은 띠, 소음은 3dB 아래부터 위쪽 전부.
    사이에 5dB 의 빈 띠가 있어 한 구간이 두 마커를 함께 받지 않는다.

    조건 넷 (2026-08-12 신설, GPT 5세션으로 보정):

    ① 강도 중앙값 ≥ 기준선 + NOISE_DB(−3.0dB) — '발화 수준에 근접'
       갭(≥0.2초) 1,755개의 기준선 대비 강도 분위는
       p50=−16.0 · p75=−7.5 · p90=+2.6 · p95=+6.0 이다. −3dB 은 대략 p87 —
       "말소리만큼 큰 갭" 상위 1할이다.

    ② 유성 프레임 비율 ≤ NOISE_VOICED_FRAC(0.10)
       챗봇 TTS 가 폰 스피커로 새어 나오는 구간이 이 코퍼스에 많은데,
       합성음도 성대 진동에 해당하는 주기성이 있어 F0 가 잡힌다. 비율로
       거르면 그 구간이 통째로 빠진다(들숨 쪽의 `len(f0) <= 1` 은 구간이
       0.15~1.2초로 짧아 개수로 충분하지만, 소음은 길 수 있어 비율을 쓴다).

    ③ 갭 양 끝에서 NOISE_EDGE(0.08초)를 잘라 낸 **속**만 재고, 그 속이
       NOISE_MIN(0.20초) 이상일 것.
       1차 실측에서 걸린 후보의 상당수가 낱말 끝 치찰음(/s/ /z/ /ks/)이
       Whisper 의 낱말 끝 타임스탬프 **밖으로** 흘러나온 것이었다 —
       "…for kids ((소음)) that there…" 처럼 갭 가장자리에 붙어 있고
       에너지의 98%가 4kHz 위였다. 그건 소음이 아니라 말소리다. 가장자리를
       버리면 세션당 8.6곳 → 4.2곳으로 줄고, 줄어든 쪽이 전부 낱말에 붙은
       조각이었다.

    ④ 중대역(500~4000Hz) 비중 ≥ NOISE_MID_DB(−6.0dB) — '광대역일 것'
       ③으로도 남는 치찰음 누출과, 마이크에 닿은 초저역 충격(에너지 99%가
       500Hz 아래, 중심주파수 11~21Hz — 사실상 들리지 않는다)을 가른다.
       실측 분리는 mid_band_ratio() 주석 참조.

    문턱별 세션당 개수 (GPT 5세션, ②=0.10 고정):

      | 여백 \\ 강도 | −6dB | −4dB | −3dB | −2dB | 0dB |
      |---|---|---|---|---|---|
      | 0.00(없음) | 16.0 | 11.0 | 8.6 | 6.8 | 5.6 |
      | 0.05       | 10.4 |  6.2 | 5.2 | 4.2 | 2.6 |
      | **0.08**   |  7.2 |  5.0 | **4.2** | 3.2 | 2.2 |
      | 0.10       |  5.8 |  4.0 | 3.6 | 3.2 | 2.2 |
      ④까지 걸면 −3dB·0.08 에서 세션당 **1.4곳**(6·0·0·0·1)이 된다.

    사람 전사본 대조 (2026-08-12, `3 전사 CA 방식 워드 파일` 10세션):
      전사자가 ((…)) 로 적은 주석은 10세션 합계 12곳(세션당 0~10, 중앙값 0)
      이고 그중 대부분이 소음이 아니라 **몸짓·행동**(touch the screen 3 ·
      look at the instruction 2 · look at the screen 1 …)이다. 소리에
      해당하는 것은 전체를 통틀어
      ((making sounds)) · ((소리)) · ((sniff)) 셋뿐이다. 즉 사람의 소음
      주석은 세션당 0~2곳이고, 자동 검출도 그 자릿수여야 한다.
      지시가 "애매하면 안 찍는 쪽"이므로 사람 상한(2곳)에 맞추기보다
      한 자릿수 안에서 가장 근거가 뚜렷한 조합을 골랐다."""
    if track is None or base_db is None:
        return []
    import statistics
    out = []
    for s, e in gaps:
        if e - s < NOISE_MIN + 2 * NOISE_EDGE or e - s > NOISE_MAX:
            continue
        if exclude and any(not (e < xs or s > xe) for xs, xe in exclude):
            continue
        cs, ce = s + NOISE_EDGE, e - NOISE_EDGE     # ③ 가장자리를 버린다
        db = track.db_in(cs, ce)
        if len(db) < 3:
            continue
        if statistics.median(db) < base_db + NOISE_DB:       # ①
            continue
        if len(track.f0_in(cs, ce)) / len(db) > NOISE_VOICED_FRAC:   # ②
            continue
        mid = track.mid_band_ratio(cs, ce)                   # ④
        if mid is None or mid < NOISE_MID_DB:
            continue
        # 근거를 마커 info 에 남긴다 — 들숨과 같은 형식에 소음의 결정 조건인
        # 중대역 비중(mid)만 덧붙였다. 검증 때 이 값이 없으면 왜 찍혔는지
        # 되짚을 수 없다(들숨 46건 청취 검증에서 실제로 겪은 문제다).
        med = statistics.median(db)
        cog = track.cog(cs, ce)
        cog_s = f"{cog:.0f}" if cog is not None else "?"
        info = (f"dur={ce - cs:.2f}s db={med - base_db:+.1f} "
                f"cog={cog_s} env={max(db) - med:.1f} mid={mid:+.1f}")
        out.append((cs, ce, info))
    return out
