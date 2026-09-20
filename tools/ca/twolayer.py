# -*- coding: utf-8 -*-
"""2층 전사 구조 — Chung & Lee 2026 방법 절을 코드로 옮긴 것.

그 논문의 실제 작업 순서(2026-08-20 확인):
  클로바노트 전체 전사 → Excel 변환 → 키워드 검색 → 전문 검증
  → 해당 구간만 사람이 Jefferson 규약으로 재전사 (3행: 한국어/글로스/자연역)

이 스크립트가 대체하는 단계:
  1층(index)   "Excel 변환 + 검색" — 전 세션 턴을 하나의 검색 CSV 로
  선별(grep/gap) "키워드 검색"     — 정규식 또는 긴 침묵 상위 N
  2층(excerpt) "재전사"의 뼈대     — 이미 gblite 가 입힌 마커를 그대로 쓰고,
                                     글로스(GLS)·자연역(ENG) 행은 비워 둔다.
                                     **사람(또는 세션 내 LLM)이 채우는 칸이다.**

로마자 행은 기본 꺼짐(--yale 로 켬). Chung & Lee 는 이 행을 뺐고("we omit
the romanized, as-pronounced line"), 한글→Yale 변환은 결정적이라 언제든
다시 붙일 수 있으므로 삭제가 아니라 토글로 둔다. 학술지에 따라 요구가
갈린다(AI & Society 는 없이 통과, ROLSI 계열 한국어 논문은 흔히 포함).

전사를 다시 하지 않는다 — 텍스트·마커는 out_aligned 산출물이 입력이다
(realign_batch 의 원칙 그대로).
"""
import argparse
import csv
import re
import sys
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SRC = ROOT / "data" / "out_aligned_260813"

# 검색층에서 지울 표기 기호 — 검색은 낱말로 하고 표기는 2층에서 되살린다.
_MARKS = re.compile(r"\(\d+\.\d+\)|\(\.\)|\(\(.*?\)\)|[↑↓°_<>=\[\]]")


def plain(text: str) -> str:
    return re.sub(r"\s+", " ", _MARKS.sub(" ", text)).strip()


def read_turns(session_dir: Path):
    """conversation_turns.csv → [(spk, raw, t0, t1)]. 화자 빈칸 행은 침묵."""
    f = session_dir / "conversation_turns.csv"
    with open(f, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [(r["SPEAKER LABEL"], r["TEXT"],
             float(r["START TIME"]), float(r["END TIME"])) for r in rows]


# ── 1층: 검색 색인 ──────────────────────────────────────────────────────
def build_index(src: Path, out_csv: Path) -> None:
    sessions = sorted(d for d in src.iterdir() if d.is_dir())
    ok, skipped, n_turns = 0, [], 0
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["session", "speaker", "t0", "t1", "plain", "raw"])
        for d in sessions:
            try:
                turns = read_turns(d)
            except FileNotFoundError:
                skipped.append(d.name)
                continue
            ok += 1
            for spk, raw, t0, t1 in turns:
                if not spk:          # 침묵 행은 색인하지 않는다 (검색 대상 아님)
                    continue
                n_turns += 1
                w.writerow([d.name, spk, f"{t0:.2f}", f"{t1:.2f}", plain(raw), raw])
    # 뺀 개수를 같은 자리에 출력한다 (CLAUDE.md 점검표).
    print(f"[1층] 세션 {ok}/{len(sessions)} 색인 · 턴 {n_turns} · 건너뜀 {len(skipped)}")
    for name in skipped:
        print(f"   건너뜀(turns.csv 없음): {name}")
    print(f"[1층] → {out_csv}")


# ── 선별 ────────────────────────────────────────────────────────────────
def select_windows(turns, grep=None, gap_top=0, ctx=2):
    """반환: [(사유, i0, i1)] — 턴 인덱스 창. 겹치면 병합."""
    hits = []
    if grep:
        pat = re.compile(grep)
        for i, (spk, raw, _, _) in enumerate(turns):
            if spk and pat.search(plain(raw)):
                hits.append((f'grep "{grep}"', i))
    if gap_top:
        sil = [(t1 - t0, i) for i, (spk, _, t0, t1) in enumerate(turns) if not spk]
        for dur, i in sorted(sil, reverse=True)[:gap_top]:
            hits.append((f"침묵 {dur:.1f}초", i))
    wins = sorted((max(0, i - ctx), min(len(turns), i + ctx + 1), why)
                  for why, i in hits)
    merged = []
    for a, b, why in wins:
        if merged and a <= merged[-1][1]:
            prev = merged[-1]
            joined = prev[2] if why in prev[2] else prev[2] + " + " + why
            merged[-1] = (prev[0], max(prev[1], b), joined)
        else:
            merged.append((a, b, why))
    total = sum(1 for spk, *_ in turns if spk)
    print(f"[선별] 후보 {len(hits)}곳 → 창 {len(merged)}개 (전체 {total}턴 중)")
    return merged


# ── Yale 로마자 (토글, 잠정) ────────────────────────────────────────────
# 근거: Martin(1992) Yale 체계. 구현은 자모 대응만 하는 잠정판 —
# 순음 뒤 ㅜ→u 규칙은 반영, 그 밖의 형태음운 축약은 하지 않는다.
_ONS = ["k", "kk", "n", "t", "tt", "l", "m", "p", "pp",
        "s", "ss", "", "c", "cc", "ch", "kh", "th", "ph", "h"]
_VOW = ["a", "ay", "ya", "yay", "e", "ey", "ye", "yey", "o", "wa", "way",
        "oy", "yo", "wu", "we", "wey", "wi", "yu", "u", "uy", "i"]
_COD = ["", "k", "kk", "ks", "n", "nc", "nh", "t", "l", "lk", "lm", "lp",
        "ls", "lth", "lph", "lh", "m", "p", "ps", "s", "ss", "ng", "c",
        "ch", "kh", "th", "ph", "h"]
_LABIAL = {"p", "pp", "ph", "m"}


def yale(text: str) -> str:
    out = []
    for ch in text:
        code = ord(ch)
        if 0xAC00 <= code <= 0xD7A3:
            i = code - 0xAC00
            ons, vow, cod = _ONS[i // 588], _VOW[i % 588 // 28], _COD[i % 28]
            if vow == "wu" and ons in _LABIAL:
                vow = "u"                      # Yale: 순음 뒤 wu → u
            out.append(ons + vow + cod)
        else:
            out.append(ch)
    return "".join(out)


# ── 2층: 3행 발췌 ──────────────────────────────────────────────────────
def excerpt(session_dir: Path, out_dir: Path, grep, gap_top, ctx, add_yale):
    turns = read_turns(session_dir)
    wins = select_windows(turns, grep=grep, gap_top=gap_top, ctx=ctx)
    if not wins:
        print("[2층] 선별된 창이 없다 — 발췌 파일을 만들지 않음")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    name = session_dir.name
    path = out_dir / f"발췌_{name}.txt"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(f"# 2층 발췌 — {name}\n")
        fh.write("# 형식: Chung & Lee 2026 3행 (원문 / GLS 직역 글로스 / ENG 자연역)\n")
        fh.write("# 원문 행 마커는 gblite 자동 측정층 그대로다. 운율 기호(↑↓°><)는\n")
        fh.write("# 자동 추정이므로 인용 전 원음 확인 필요. GLS·ENG 는 빈칸 = 사람 몫.\n")
        if add_yale:
            fh.write("# YAL 행: Yale 로마자 (잠정 구현 — 자모 대응만, 인용 전 검수)\n")
        for k, (a, b, why) in enumerate(wins, 1):
            t0, t1 = turns[a][2], turns[b - 1][3]
            fh.write(f"\n── 발췌 {k} · {t0:.1f}–{t1:.1f}초 · 사유: {why} ──\n")
            for spk, raw, u0, u1 in turns[a:b]:
                if not spk:
                    fh.write(f"      {raw}\n")     # 침묵 행 그대로
                    continue
                fh.write(f"{spk}  {raw}  [{u0:.2f}–{u1:.2f}]\n")
                if add_yale:
                    fh.write(f" YAL  {yale(plain(raw))}\n")
                fh.write(" GLS  \n ENG  \n")
    print(f"[2층] → {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p1 = sub.add_parser("index", help="1층: 전 세션 검색 색인")
    p1.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p1.add_argument("--out", type=Path,
                    default=ROOT / "reports" / "twolayer_260820" / "layer1_index.csv")
    p2 = sub.add_parser("excerpt", help="2층: 선별 구간 3행 발췌")
    p2.add_argument("--session", required=True)
    p2.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p2.add_argument("--out", type=Path,
                    default=ROOT / "reports" / "twolayer_260820")
    p2.add_argument("--grep", default=None, help="선별: 정규식 (마커 제거 후 매칭)")
    p2.add_argument("--gap-top", type=int, default=0, help="선별: 긴 침묵 상위 N")
    p2.add_argument("--ctx", type=int, default=2, help="창 확장 ±턴 수 (잠정)")
    p2.add_argument("--yale", action="store_true", help="Yale 로마자 행 추가")
    a = ap.parse_args()
    if a.cmd == "index":
        build_index(a.src, a.out)
    else:
        d = a.src / a.session
        if not d.is_dir():
            sys.exit(f"세션 없음: {d}")
        excerpt(d, a.out, a.grep, a.gap_top, a.ctx, a.yale)


if __name__ == "__main__":
    main()
