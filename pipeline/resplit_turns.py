#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""conversation_turns.csv 의 뭉친 턴을 낱말 간격으로 다시 쪼개는 **후처리 프로토타입**.

왜 필요한가 (2026-08-12 실측)
---------------------------
gblite 의 턴 경계는 사실상 **화자 교대 하나뿐**이다.

  1) analysis.build_utterances() 는 같은 화자의 낱말을
     TURN_END_THRESHOLD_SECS(기본 0.1초, measured 프리셋 0.45초) 미만 침묵으로만
     묶어 발화(utterance)를 만든다. 여기까지는 잘게 쪼개진다.
  2) analysis.detect_pauses() 가 같은 화자의 인접 발화 사이 침묵에
     latch / micropause / pause / **largepause** 표시를 단다. largepause 는
     `fto >= LB_LARGE_PAUSE(1.0)` 이면 무조건이라 **상한이 없다** — 80초 침묵도
     largepause 하나로 표시된다.
  3) output.assemble() 은 "같은 화자 + 경계에 pause 계열 표시가 있으면 병합"이라
     2)에서 붙인 largepause 때문에 침묵 길이와 무관하게 계속 이어 붙인다.

결과적으로 같은 화자가 (중간에 다른 화자가 끼어들지 않는 한) 몇 분을 쉬든
한 턴으로 뭉친다. CB003 은 SP_B 가 870초짜리 턴 하나(전체 975초의 89%)로
나왔고, 4인 토론인 88-002 도 턴이 14개뿐이었다.

이 스크립트가 하는 일
--------------------
conversation_words.csv(낱말 단위 타임스탬프)를 읽어 기존 턴 **안에서**
같은 화자의 낱말 간격이 임계값(기본 1.0초) 이상인 자리를 턴 경계로 삼는다.
- 쪼개진 자리의 침묵은 CA 표기로 되살린다: 원본 CSV 의 gap 행 관례를 따라
  화자 칸이 빈 `(x.x)` 행을 두 턴 사이에 넣는다.
- 임계값 미만이라 안 쪼갠 침묵은 낱말 사이에 인라인으로 되돌린다
  (0.2초 이상 `(x.x)`, 0.1~0.2초 `(.)` — gblite analysis.py 의 문턱과 동일).
- 쪼갤 자리가 없는 턴은 **원본 행을 그대로 둔다**. 재구성하면 CA 표기가
  날아가므로, 손대야 할 턴만 손대는 게 손실이 적다.

★ 한계 — 이건 프로토타입이지 수정이 아니다 ★
--------------------------------------------
1. **CA 표기 소실**: 원본 turns 의 TEXT 에는 억양(∇ ∆), 겹침 대괄호 [ ],
   래칭 =, 늘임 :::, 크기/속도 표시(> < ° °), ((소음)) 등이 들어 있는데,
   낱말 CSV 에는 낱말과 시각밖에 없다. **쪼개진 턴의 텍스트에서는 이 표기가
   전부 사라진다.** 침묵 표기만 복원된다.
2. **겹침 구간 경계가 부정확**: 낱말 CSV 는 시간순 단일 열이라 두 화자가
   겹쳐 말한 구간의 소속을 정확히 되살릴 수 없다.
3. **낱말 끝시각이 늘어난다**: Whisper 계열 낱말 타임스탬프는 낱말 끝을
   다음 낱말 시작까지 늘려 잡는 경향이 있어, 여기서 재는 간격은 실제 침묵보다
   **짧게** 나온다. 즉 임계값 1.0초는 실제 침묵 1.0초보다 관대하다.
4. **화자 배정은 손대지 않는다**: 진단 결과 CB003 은 다이어리제이션 자체가
   챗봇/사람을 뭉뚱그린 정황이 있는데, 이 스크립트는 화자 라벨을 그대로 믿는다.

→ **진짜 수정은 gblite 파이프라인 안에서 해야 한다.** 구체적으로는
   output.assemble() 의 병합 조건에 "침묵 상한"을 넣거나
   (예: largepause 가 MAX_INTRA_TURN_SILENCE 이상이면 병합하지 않고 턴을 끊는다),
   analysis.detect_pauses() 에서 상한 없는 largepause 를 턴 경계 후보로 분리하는
   식이어야 한다. 그래야 CA 표기·겹침·억양이 살아 있는 채로 턴이 쪼개진다.
   이 후처리는 내일 미팅용 수치를 뽑기 위한 임시방편이다.

사용법
------
  python3 resplit_turns.py <결과폴더> [--gap 1.0] [--out 파일명]
  # 출력: <결과폴더>/conversation_turns_resplit.csv (원본은 건드리지 않는다)
"""
import argparse
import csv
import os
import sys

HEADER = ["SPEAKER LABEL", "TEXT", "START TIME", "END TIME"]

# 인라인 침묵 표기 문턱 — gblite/analysis.py Thresholds 와 맞춘다.
LB_MICROPAUSE = 0.1     # 이 미만은 표기하지 않는다
LB_PAUSE = 0.2          # 이 이상은 (x.x), 그 아래는 (.)
EPS = 0.005             # 부동소수 비교용 여유


def read_rows(path):
    """CSV 를 dict 리스트로 읽는다. BOM 이 붙은 파일도 견디게 utf-8-sig."""
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_words(path):
    """낱말 CSV → (화자, 시작, 끝, 텍스트) 리스트. 시간순 정렬."""
    words = []
    for r in read_rows(path):
        spk = (r.get("SPEAKER LABEL") or "").strip()
        txt = (r.get("TEXT") or "").strip()
        if not spk or not txt:
            continue
        try:
            s, e = float(r["START TIME"]), float(r["END TIME"])
        except (TypeError, ValueError):
            continue
        words.append((spk, s, e, txt))
    words.sort(key=lambda w: (w[1], w[2]))
    return words


def pause_symbol(gap):
    """CA 침묵 표기. 소수 첫째 자리까지만 — 귀로 0.32와 0.3을 구분 못 하므로
    둘째 자리는 정밀도를 과장하는 표기다(gblite 와 동일한 결정)."""
    if gap < LB_PAUSE:
        return "(.)"
    return "(%.1f)" % gap


def split_chunk(chunk, gap_thr):
    """한 턴에 속한 낱말들을 간격 >= gap_thr 자리에서 토막 낸다.

    반환: [[(spk,s,e,txt), ...], ...] — 토막마다 낱말 리스트."""
    pieces = [[chunk[0]]]
    for prev, cur in zip(chunk, chunk[1:]):
        if cur[1] - prev[2] >= gap_thr - EPS:
            pieces.append([cur])          # 여기서 턴을 끊는다
        else:
            pieces[-1].append(cur)
    return pieces


def render(piece):
    """낱말 토막 → 텍스트. 남은 침묵은 낱말 사이에 CA 표기로 되돌린다.

    주의: 억양·겹침·늘임 등 원본 CA 표기는 낱말 CSV 에 없어 복원 불가다."""
    out = [piece[0][3]]
    for prev, cur in zip(piece, piece[1:]):
        gap = cur[1] - prev[2]
        if gap >= LB_MICROPAUSE - EPS:
            out.append(pause_symbol(gap))
        out.append(cur[3])
    return " ".join(out)


def resplit(folder, gap_thr, out_name="conversation_turns_resplit.csv"):
    turns_path = os.path.join(folder, "conversation_turns.csv")
    words_path = os.path.join(folder, "conversation_words.csv")
    if not (os.path.isfile(turns_path) and os.path.isfile(words_path)):
        raise FileNotFoundError("conversation_turns.csv / conversation_words.csv 없음: %s" % folder)

    rows = read_rows(turns_path)
    words = load_words(words_path)

    new_rows = []
    n_src_turns = n_out_turns = 0
    for r in rows:
        spk = (r.get("SPEAKER LABEL") or "").strip()
        if not spk:
            # 화자 칸이 빈 행 = 원본의 gap / ((소음)) 행. 그대로 흘려보낸다.
            new_rows.append([spk, r.get("TEXT", ""),
                             r.get("START TIME", ""), r.get("END TIME", "")])
            continue
        n_src_turns += 1
        try:
            ts, te = float(r["START TIME"]), float(r["END TIME"])
        except (TypeError, ValueError):
            new_rows.append([spk, r.get("TEXT", ""),
                             r.get("START TIME", ""), r.get("END TIME", "")])
            n_out_turns += 1
            continue

        # 이 턴의 시간창 안에 있는 같은 화자의 낱말만 가져온다.
        chunk = [w for w in words
                 if w[0] == spk and w[1] >= ts - EPS and w[2] <= te + EPS]
        pieces = split_chunk(chunk, gap_thr) if chunk else []

        if len(pieces) <= 1:
            # 쪼갤 자리가 없다 → 원본 텍스트를 그대로 둔다.
            # 재구성하면 CA 표기(억양·겹침)가 날아가니, 건드리지 않는 편이 낫다.
            new_rows.append([spk, r.get("TEXT", ""),
                             "%.2f" % ts, "%.2f" % te])
            n_out_turns += 1
            continue

        for i, piece in enumerate(pieces):
            ps, pe = piece[0][1], piece[-1][2]
            if i == 0:
                ps = ts                     # 턴 앞머리 무음은 원본 경계를 지킨다
            if i == len(pieces) - 1:
                pe = te
            if i > 0:
                # 쪼갠 자리의 침묵을 잃지 않게 gap 행으로 되살린다
                # (화자 칸을 비우는 것은 원본 CSV 의 gap 행 관례와 같다).
                gs, ge = pieces[i - 1][-1][2], piece[0][1]
                new_rows.append(["", "(%.1f)" % (ge - gs),
                                 "%.2f" % gs, "%.2f" % ge])
            new_rows.append([spk, render(piece), "%.2f" % ps, "%.2f" % pe])
            n_out_turns += 1

    out_path = os.path.join(folder, out_name)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADER)
        w.writerows(new_rows)
    return out_path, n_src_turns, n_out_turns


def turn_stats(rows):
    """(턴수, 최장초, 60초초과턴수) — 화자 칸이 빈 gap/소음 행은 세지 않는다."""
    durs = []
    for r in rows:
        if not (r.get("SPEAKER LABEL") or "").strip():
            continue
        try:
            durs.append(float(r["END TIME"]) - float(r["START TIME"]))
        except (TypeError, ValueError):
            continue
    if not durs:
        return 0, 0.0, 0
    return len(durs), max(durs), sum(1 for d in durs if d > 60)


def main():
    ap = argparse.ArgumentParser(
        description="conversation_turns.csv 를 낱말 간격으로 재분할(후처리 프로토타입)")
    ap.add_argument("folder", help="세션 결과 폴더 (conversation_*.csv 가 있는 곳)")
    ap.add_argument("--gap", type=float, default=1.0,
                    help="턴을 끊을 낱말 간격 임계값(초). 기본 1.0")
    ap.add_argument("--out", default="conversation_turns_resplit.csv",
                    help="출력 파일명(폴더 안). 원본은 절대 덮어쓰지 않는다")
    args = ap.parse_args()

    if os.path.abspath(args.out) == os.path.abspath(
            os.path.join(args.folder, "conversation_turns.csv")):
        sys.exit("원본 conversation_turns.csv 는 덮어쓸 수 없다")

    folder = os.path.abspath(args.folder)
    before = read_rows(os.path.join(folder, "conversation_turns.csv"))
    out_path, _, _ = resplit(folder, args.gap, args.out)
    after = read_rows(out_path)

    b = turn_stats(before)
    a = turn_stats(after)
    print("[%s] 임계값 %.1fs" % (os.path.basename(folder), args.gap))
    print("  before: 턴 %d개 · 최장 %.1fs · 60s초과 %d개" % b)
    print("  after : 턴 %d개 · 최장 %.1fs · 60s초과 %d개" % a)
    print("  →", out_path)


if __name__ == "__main__":
    main()
