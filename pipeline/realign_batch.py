# -*- coding: utf-8 -*-
"""기존 산출물을 **재전사 없이** 정렬만 다시 해서 새로 조립한다.

왜 재전사가 필요 없는가 — 1.9절 원리 그대로다. "전사를 다시 하지 않는다.
텍스트는 입력이지 출력이 아니다." `align_words()` 는 낱말 목록과 음원만 있으면
시간을 다시 잡는다. 기존 산출물의 `conversation_words.csv` 가 그 낱말 목록이다.

**`utt.toml` 이 아니라 CSV 를 읽는 이유** (2026-08-13 실측):
  · `gailbot_results` 계열은 2026-08-06 수정 **이전**에 쓰여 한글 테이블명이
    따옴표 없이 들어갔다 → tomllib 가 못 읽는다("Expected ']]'").
  · `_v2` 계열은 읽히지만 화자 라벨이 `SP_SP_SP_A` 로 접두어가 세 번 붙었다.
  CSV 는 두 문제 모두 없다. 화자 라벨은 그래도 정규화해 둔다.

산출은 **새 디렉토리**에 쓴다. 기존 전사본을 덮지 않아야 정렬 전/후를 비교할
수 있다(측정값_보존_260812_pre_align.md).

정렬된 낱말은 새 `utt.toml` 로 남는다. 이후 문턱·턴 규칙을 바꾸면 `annotate`
만 다시 돌리면 되고 재정렬이 필요 없다 — 2.24절에서 정렬 결과를 안 남겨
7건 중 5건을 다시 정렬해야 했던 선례를 피한다.

사용:
  python realign_batch.py <결과폴더> [결과폴더...] -o <출력루트> [--preset aligned]
  python realign_batch.py --dry-run <결과폴더>      # 대상만 세어 본다
"""
import argparse
import csv
import os
import re
import sys
import time
import unicodedata

# 하위 폴더에서 실행해도 gblite 를 찾게 한다 (2026-08-13 폴더 정리 후).
# 파이썬은 스크립트가 있는 폴더를 sys.path[0] 으로 잡으므로,
# diag/ · pipeline/ 등에서 돌리면 프로젝트 루트가 경로에 없다.
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

from gblite import align, analysis, output
from gblite.models import Word

N = lambda s: unicodedata.normalize("NFC", s)
AUD_EXT = (".wav", ".mp3", ".m4a", ".mp4", ".mts", ".mkv")


def index_audio(roots):
    """이름 → 음원 경로. 유니코드 정규화를 맞춰야 한글 파일명이 걸린다."""
    idx = {}
    for root in roots:
        for dirpath, _, files in os.walk(root):
            if dirpath.count(os.sep) > 7:
                continue
            for f in files:
                if f.lower().endswith(AUD_EXT):
                    idx.setdefault(N(os.path.splitext(f)[0]),
                                   os.path.join(dirpath, f))
    return idx


def load_words(csv_path):
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    if not rows:
        return []
    ks = [k for k in rows[0] if "START" in k.upper()][0]
    ke = [k for k in rows[0] if "END" in k.upper()][0]
    kt = [k for k in rows[0] if "TEXT" in k.upper() or "WORD" in k.upper()][0]
    ksp = next((k for k in rows[0] if "SPEAK" in k.upper()), None)
    out = []
    for r in rows:
        try:
            s, e = float(r[ks]), float(r[ke])
        except (ValueError, KeyError):
            continue
        # SP_SP_SP_A 처럼 접두어가 겹쳐 붙은 라벨을 정규화한다.
        spk = re.sub(r"^(SP_)+", "", (r.get(ksp) or "").strip()) if ksp else ""
        out.append(Word(start=s, end=e, text=r[kt], speaker=spk or "A"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folders", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--preset", default="aligned")
    ap.add_argument("--audio-root", default=os.path.expanduser("~/Downloads"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.dry_run and not a.out:
        sys.exit("-o 출력루트가 필요하다 (기존 산출물을 덮지 않는다)")

    aud = index_audio([a.audio_root])
    print(f"음원 색인 {len(aud)}개")

    jobs, seen, skipped = [], {}, []
    for folder in a.folders:
        for d in sorted(os.listdir(folder)):
            sub = os.path.join(folder, d)
            wcsv = os.path.join(sub, "conversation_words.csv")
            if not os.path.isdir(sub) or not os.path.exists(wcsv):
                continue
            name = N(d)
            audio = aud.get(name)
            if not audio:
                skipped.append((name, "음원 없음"))
                continue
            # 같은 세션이 여러 폴더에 있으면 **낱말이 많은 판본**을 쓴다.
            # 판본 간 차이는 화자 분리 재실행에서 온다(2026-08-13 확인).
            n_words = sum(1 for _ in open(wcsv, encoding="utf-8")) - 1
            if name in seen and seen[name][0] >= n_words:
                continue
            seen[name] = (n_words, sub, audio)
    jobs = [(k, v[1], v[2], v[0]) for k, v in sorted(seen.items())]

    print(f"대상 {len(jobs)}세션 (중복 제거 후) · 건너뜀 {len(skipped)}")
    if a.dry_run:
        for n, sub, audio, nw in jobs[:8]:
            print(f"  {n[:34]:34s} 낱말 {nw:5d}  ← {os.path.basename(os.path.dirname(sub))}")
        print(f"  ... 총 {len(jobs)}")
        return

    th = analysis.THRESHOLD_PRESETS[a.preset]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    done = fail = 0
    for i, (name, sub, audio, nw) in enumerate(jobs, 1):
        outdir = os.path.join(a.out, name)
        if os.path.exists(os.path.join(outdir, "conversation.txt")):
            print(f"[{i}/{len(jobs)}] {name[:32]} — 건너뜀(이미 있음)", flush=True)
            continue
        try:
            words = load_words(os.path.join(sub, "conversation_words.csv"))
            if not words:
                raise ValueError("낱말 없음")
            t1 = time.time()
            aligned = align.align_words(words, audio)
            utts, markers, stats = analysis.analyze(
                aligned, th=th, audio_path=audio)
            # 파일명을 기존 산출물과 맞춘다. name 을 그대로 주면
            # "<세션명>_words.csv" 가 되어 하위 분석 스크립트가 찾는
            # conversation_words.csv 와 어긋난다(2026-08-13).
            output.write_all(outdir, "conversation", utts, markers,
                             {name: aligned}, stats, lang=None)
            done += 1
            print(f"[{i}/{len(jobs)}] {name[:32]:32s} 낱말 {len(words):5d} "
                  f"발화 {len(utts):4d} 마커 {len(markers):5d} "
                  f"{time.time()-t1:5.1f}초", flush=True)
        except Exception as e:
            fail += 1
            print(f"[{i}/{len(jobs)}] {name[:32]:32s} 실패: "
                  f"{type(e).__name__} {e}", flush=True)
    print(f"\n완료 {done} · 실패 {fail} · 소요 {(time.time()-t0)/60:.1f}분")


if __name__ == "__main__":
    main()
