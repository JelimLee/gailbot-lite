# -*- coding: utf-8 -*-
"""배치 산출물 품질 관리 — **조용한 실패를 잡는다.**

같은 유형의 실패가 세 번 났다(2026-08-13 정리).

  1.6절   parselmouth 미설치 → 운율이 통째로 빠짐
  1.10-④  laughter 의 transformers 미설치 → 같은 패턴
  오늘    parselmouth 가 MTS 를 못 읽음 → `except Exception: return None` 이 삼킴

앞의 둘은 **부재**만 표면화했고 런타임 실패는 그대로 남아 있었다. 교훈을
넓힌다 — **부재든 실패든, 기능이 조용히 줄어들면 안 된다.**

`align.align_words()` 도 같은 방침이다(1.9절): whisperx 미설치·모델 적재
실패·음원 열기 실패가 전부 **원본 타임스탬프 반환**으로 빠진다. 그리고 그것은
성공과 구별되지 않는다. whisperx 의 "backtrack failed" 도 조용히 원본으로
되돌린다.

그래서 결과물 쪽에서 역으로 검출한다. **낱말 간격이 정확히 0인 비율**이
완벽한 탐지기다.

    0.4% 근처   정렬 성공
    95% 근처    정렬이 조용히 실패했다

중간이 거의 없다 — 되거나 안 되거나이기 때문이다.

2026-08-21 에 **네 번째**가 났다. `scan_hallucination.py` 가 경로가 어긋나
환각 0건을 내고 있었는데 이 스크립트가 그것을 못 봤다 — 환각이 점검 항목에
아예 없었기 때문이다(`grep -n "환각|hallu|compression|logprob" batch_qc.py` → 0건).
고칠 곳은 스캐너가 아니라 여기였다. 그래서 두 항목을 더한다.

  (ㄱ) 세그먼트 신뢰도 — `conversation_segments.csv` 의 `RADFORD SUSPECT`.
       파일이 없으면 **"의심 0건"이 아니라 "이 세션은 못 쟀음"** 이다.
       `unknown` 도 "멀쩡함"이 아니라 "못 쟀음"이다. 셋을 같은 칸에 찍지 않는다.
  (ㄴ) 반복 붕괴 — 낱말 CSV(측정층)를 30초 창으로 묶어 zlib 압축률을 잰다.
       턴 CSV 는 CA 마커가 섞인 표기층이라 쓰지 않는다.

**세기만 한다. 아무것도 지우거나 고치지 않는다.**

사용:
  python batch_qc.py <배치출력루트> [--log 배치로그] [--csv 저장경로]
"""
import argparse
import csv
import glob
import os
import re
import sys
import unicodedata
import zlib
from collections import Counter

N = lambda s: unicodedata.normalize("NFC", s)

# 정렬 성공/실패를 가르는 경계. 실측 분포가 양극단이라 어디를 잡아도 같지만,
# 5% 는 "성공한 세션의 가장 나쁜 값"에도 여유를 둔 값이다.
ZERO_GAP_FAIL = 5.0
PROSODY_KINDS = ("intonation", "volume", "stress", "breath", "laugh")

# ── 반복 붕괴(같은 말이 반복돼 텍스트가 지나치게 잘 압축된다) ──────────────
# 창 길이. 원 논문이 Whisper 세그먼트(최대 30초)를 단위로 삼았으므로 맞춘다.
WINDOW_SEC = 30.0
#
# 문턱값 근거:
#   · Radford et al. 2022 §4.5 의 원 문턱은 compression_ratio > 2.4 다 `문헌`.
#   · 그런데 이 코퍼스(한국어 L2, 한/영 혼합)에서는 그 값이 너무 높다.
#     58세션을 30초 창 3,673개로 묶어 전수 재계산해 보니 **음성 창 3,544개의
#     최대가 2.34** 라 2.4 는 한 창도 잡지 못한다. 최적 F1 문턱은 **2.02**
#     (정밀도 0.278)였다.
#     `실측(58세션·30초 창 3,673개·건너뜀 0, 2026-08-21)`
#   · 다만 그 실측은 **텍스트 스캐너 후보와 Whisper 지표의 일치도**이지 확정
#     환각과의 대조가 아니다. 96곳은 아직 한 곳도 듣지 않았다. 그리고 신호가
#     있는 것은 반복 유형뿐이고(AP 0.091 = 기준선의 7.5배), 조음 유형은
#     ROC-AUC 0.388 로 우연 이하다 — **이 지표로 조음을 가르려 하지 마라.**
#   · 그러므로 2.02 는 `잠정` 이다. 사람 대조 전까지 딱지를 떼지 않는다.
#     선별 도구가 아니라 **우선순위 매기기**용이다(문턱 2.4 의 재현율은 4.5%).
COMPRESSION_ALERT = 2.02          # 잠정 — 위 근거 참조
COMPRESSION_RADFORD = 2.4         # 문헌: Radford et al. 2022 §4.5 (참고로 같이 셈)


def hms(sec):
    """초 → m:ss. 사람이 음원에서 찾아 들을 수 있는 형식으로 찍는다."""
    return f"{int(sec) // 60}:{int(sec) % 60:02d}"


def compression_ratio(text):
    """zlib 압축률. faster_whisper 1.2.1 `transcribe.py:1879` 의
    `get_compression_ratio` 와 같은 식이다(텍스트만 받는다 — 실측으로 확인).
    빈 텍스트는 잴 수 없으므로 0 이 아니라 None 을 돌려준다."""
    b = text.encode("utf-8")
    if not b:
        return None
    return len(b) / len(zlib.compress(b))


def segment_qc(d):
    """세그먼트 신뢰도 CSV 를 센다. **없으면 0 이 아니라 '못 쟀음'이다.**

    돌려주는 dict 의 `state` 가 "ok" 가 아니면 숫자 칸은 채우지 않는다.
    0 과 미측정을 같은 칸에 찍으면 다음 사람이 반드시 0 으로 읽는다.
    """
    p = os.path.join(d, "conversation_segments.csv")
    if not os.path.exists(p):
        return {"state": "파일없음"}
    rows = list(csv.DictReader(open(p, encoding="utf-8")))
    if not rows:
        return {"state": "빈파일"}
    key = next((k for k in rows[0] if "SUSPECT" in k.upper()), None)
    if key is None:
        return {"state": "열없음"}
    rkey = next((k for k in rows[0] if "REASON" in k.upper()), None)
    vals = [(r.get(key) or "").strip() for r in rows]
    n = len(vals)
    susp = sum(1 for v in vals if v == "1")
    unk = sum(1 for v in vals if v == "unknown")
    clean = sum(1 for v in vals if v == "0")
    other = n - susp - unk - clean      # 예상 밖 값. 0(멀쩡함)에 섞지 않는다
    cr = lp = 0
    if rkey:
        cr = sum(1 for r in rows if "compression_ratio" in (r.get(rkey) or ""))
        lp = sum(1 for r in rows if "avg_logprob" in (r.get(rkey) or ""))
    return {"state": "ok", "n": n, "suspect": susp, "unknown": unk,
            "clean": clean, "other": other, "cr": cr, "lp": lp}


def window_qc(rows, ks, kt):
    """낱말 CSV(측정층)를 30초 창으로 묶어 압축률을 잰다.

    턴 CSV 는 쓰지 않는다 — CA 마커가 섞인 표기층이라 압축률이 마커를 잰다.
    창을 버릴 때는 **버린 개수를 같이 돌려준다**(빈 창은 0 이 아니라 미측정).

    문턱 초과 **개수**만으로는 쓸 수 없다. 이 지표는 선별기가 아니라
    정렬기이므로(상위 5창 정밀도 0.40, 상위 100창 0.09 —
    `실측(58세션·30초 창 3,673개, 2026-08-21)`), 사람이 **들을 순서**가
    산출물이다. 그래서 압축률 상위 3창의 **시각**을 같이 돌려준다.
    """
    if kt is None:
        return {"state": "텍스트열없음"}
    bins = {}
    for r in rows:
        try:
            t = float(r[ks])
        except (TypeError, ValueError):
            bins.setdefault("bad", []).append("")
            continue
        bins.setdefault(int(t // WINDOW_SEC), []).append((r.get(kt) or "").strip())
    bad = len(bins.pop("bad", []))
    wins, empty = [], 0
    for k in sorted(bins):
        txt = " ".join(x for x in bins[k] if x)
        c = compression_ratio(txt)
        if c is None:                    # 낱말은 있는데 텍스트가 전부 빈 창
            empty += 1
            continue
        wins.append((k * WINDOW_SEC, c))
    if not wins:
        return {"state": "창없음", "empty": empty, "bad_time": bad}
    ratios = [c for _, c in wins]
    top = sorted(wins, key=lambda x: -x[1])[:3]
    return {"state": "ok", "n": len(ratios), "empty": empty, "bad_time": bad,
            "hi": sum(1 for c in ratios if c > COMPRESSION_ALERT),
            "hi_radford": sum(1 for c in ratios if c > COMPRESSION_RADFORD),
            "max": max(ratios), "top": top}


def session_qc(d):
    w = os.path.join(d, "conversation_words.csv")
    m = os.path.join(d, "conversation_markers.csv")
    cha = os.path.join(d, "conversation.cha")
    if not os.path.exists(w):
        return None
    rows = list(csv.DictReader(open(w, encoding="utf-8")))
    if not rows:
        return None
    ks = [k for k in rows[0] if "START" in k.upper()][0]
    ke = [k for k in rows[0] if "END" in k.upper()][0]
    ksp = next((k for k in rows[0] if "SPEAK" in k.upper()), None)
    kt = next((k for k in rows[0] if k.strip().upper() == "TEXT"), None)
    words = sorted((float(r[ks]), float(r[ke])) for r in rows)
    gaps = [s1 - e0 for (s0, e0), (s1, e1) in zip(words, words[1:])]
    zero = 100.0 * sum(1 for g in gaps if g == 0) / max(len(gaps), 1)

    pros = 0
    nmark = 0
    if os.path.exists(m):
        mrows = list(csv.DictReader(open(m, encoding="utf-8")))
        nmark = len(mrows)
        if mrows:
            kk = [x for x in mrows[0]
                  if "MARKER" in x.upper() or "TYPE" in x.upper()
                  or "NAME" in x.upper()][0]
            pros = sum(1 for r in mrows
                       if any(t in r[kk] for t in PROSODY_KINDS))
    lang = ""
    if os.path.exists(cha):
        for line in open(cha, encoding="utf-8"):
            if line.startswith("@Languages"):
                lang = line.split("\t")[-1].strip()
                break
    spk = len({r[ksp] for r in rows if ksp and r[ksp].strip()}) if ksp else 0

    seg = segment_qc(d)
    win = window_qc(rows, ks, kt)
    out = {"name": N(os.path.basename(d.rstrip("/"))),
           "words": len(rows), "zero_gap": zero, "markers": nmark,
           "prosody": pros, "speakers": spk, "lang": lang}
    # 못 쟀으면 숫자 칸을 빈칸으로 남긴다. 0 으로 메우지 않는다.
    out["seg_state"] = seg["state"]
    for k in ("n", "suspect", "unknown", "clean", "other", "cr", "lp"):
        out["seg_" + k] = seg.get(k, "")
    out["win_state"] = win["state"]
    for k in ("n", "empty", "bad_time", "hi", "hi_radford"):
        out["win_" + k] = win.get(k, "")
    out["win_max"] = win.get("max", "")
    # 사람이 들을 자리. CSV 에도 시각을 실어 둔다(창 시작 초 · 압축률).
    out["win_top"] = win.get("top", [])
    out["win_top_str"] = " | ".join(f"{hms(t)}={c:.2f}"
                                    for t, c in win.get("top", []))
    return out


def parse_log(path):
    """세션별 backtrack 경고 수. 로그가 없으면 빈 dict."""
    if not path or not os.path.exists(path):
        return {}
    out, cur = {}, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = re.match(r"\[\d+/\d+\]\s+(.+?)\s{2,}낱말", line)
        if m:
            cur = N(m.group(1).strip())
            out.setdefault(cur, 0)
        elif "backtrack failed" in line and cur:
            out[cur] = out.get(cur, 0) + 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--log", default=None)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()

    bt = parse_log(a.log)
    rows = []
    skipped = []                        # 뺀 폴더는 개수를 찍는다. 조용히 빼지 않는다
    for d in sorted(glob.glob(os.path.join(a.root, "*/"))):
        r = session_qc(d)
        if r:
            r["backtrack"] = bt.get(r["name"], 0)
            rows.append(r)
        else:
            skipped.append(N(os.path.basename(d.rstrip("/"))))
    if not rows:
        sys.exit(f"검사할 세션이 없다 (폴더 {len(skipped)}개를 열었지만 "
                 f"conversation_words.csv 가 없거나 비었다) — root={a.root}")

    fail_align = [r for r in rows if r["zero_gap"] >= ZERO_GAP_FAIL]
    fail_pros = [r for r in rows if r["prosody"] == 0]
    fail_lang = [r for r in rows if r["lang"] != "kor, eng"]

    print(f"세션 {len(rows)}개"
          + (f" · 낱말 CSV 가 없어 뺀 폴더 {len(skipped)}개" if skipped else "")
          + "\n")
    if skipped:
        print("  뺀 폴더: " + ", ".join(s[:28] for s in skipped[:10])
              + (" …" if len(skipped) > 10 else "") + "\n")
    print(f"{'세션':36s} {'낱말':>6s} {'간격0%':>7s} {'운율':>5s} "
          f"{'마커':>6s} {'화자':>4s} {'bt':>3s} {'의심':>6s} {'미측':>6s} "
          f"{'창':>5s} {'압축↑':>5s} {'최대압축':>8s}")
    for r in sorted(rows, key=lambda x: -x["zero_gap"]):
        flag = "  ❌" if r["zero_gap"] >= ZERO_GAP_FAIL else ""
        if r["prosody"] == 0:
            flag += " 운율0"
        if r["seg_state"] != "ok":
            s_susp = s_unk = f"{'못쟀음':>6s}"
        else:
            s_susp = f"{r['seg_suspect']:6d}"
            s_unk = f"{r['seg_unknown']:6d}"
        if r["win_state"] != "ok":
            s_n = s_hi = f"{'못쟀':>5s}"
            s_max = f"{'—':>8s}"
        else:
            s_n = f"{r['win_n']:5d}"
            s_hi = f"{r['win_hi']:5d}"
            s_max = f"{r['win_max']:8.2f}"
            if r["win_hi"]:
                flag += f" 반복?{r['win_hi']}창"
        print(f"{r['name'][:36]:36s} {r['words']:6d} {r['zero_gap']:6.1f}% "
              f"{r['prosody']:5d} {r['markers']:6d} {r['speakers']:4d} "
              f"{r['backtrack']:3d} {s_susp} {s_unk} {s_n} {s_hi} {s_max}"
              f"{flag}")

    print(f"\n── 판정 ──")
    print(f"정렬 실패 (간격0% ≥ {ZERO_GAP_FAIL:g}%): {len(fail_align)}세션")
    for r in fail_align[:10]:
        print(f"    {r['name'][:44]}  {r['zero_gap']:.1f}%")
    print(f"운율 0건: {len(fail_pros)}세션")
    for r in fail_pros[:10]:
        print(f"    {r['name'][:44]}")
    print(f"@Languages 규약 위반: {len(fail_lang)}세션")
    tot_bt = sum(r["backtrack"] for r in rows)
    print(f"backtrack failed 경고 합계: {tot_bt}건 "
          f"(경고가 난 세션 {sum(1 for r in rows if r['backtrack']):d}개)")

    ok = [r["zero_gap"] for r in rows if r["zero_gap"] < ZERO_GAP_FAIL]
    if ok:
        print(f"\n성공 세션의 간격0%: 중앙 {sorted(ok)[len(ok)//2]:.2f}% · "
              f"최대 {max(ok):.2f}%")

    # ── (ㄱ) 세그먼트 신뢰도 ────────────────────────────────────────────
    print(f"\n── 세그먼트 신뢰도 (conversation_segments.csv) ──")
    measured = [r for r in rows if r["seg_state"] == "ok"]
    unmeasured = Counter(r["seg_state"] for r in rows if r["seg_state"] != "ok")
    print(f"못 쟀음: {len(rows) - len(measured)}/{len(rows)}세션 "
          + (" · ".join(f"{k} {v}세션" for k, v in unmeasured.items())
             if unmeasured else "")
          + "   ← 이건 '의심 0건'이 아니다")
    if measured:
        tn = sum(r["seg_n"] for r in measured)
        ts = sum(r["seg_suspect"] for r in measured)
        tu = sum(r["seg_unknown"] for r in measured)
        to = sum(r["seg_other"] for r in measured)
        pct = f"{ts / tn:.1%}" if tn else "—"
        print(f"쟀음: {len(measured)}세션 · 세그먼트 {tn}개 · "
              f"의심 {ts}개({pct}, 분모 {tn}) · "
              f"판정불가(unknown) {tu}개 · 예상밖 값 {to}개")
        print(f"    사유별: compression_ratio {sum(r['seg_cr'] for r in measured)}개 · "
              f"avg_logprob {sum(r['seg_lp'] for r in measured)}개")
        for r in sorted(measured, key=lambda x: -x["seg_suspect"])[:10]:
            if r["seg_suspect"]:
                print(f"    {r['name'][:44]}  의심 {r['seg_suspect']}"
                      f"/{r['seg_n']}")

    # ── (ㄴ) 반복 붕괴 ─────────────────────────────────────────────────
    print(f"\n── 반복 붕괴 (낱말 CSV · {WINDOW_SEC:g}초 창 · zlib 압축률) ──")
    wok = [r for r in rows if r["win_state"] == "ok"]
    wbad = Counter(r["win_state"] for r in rows if r["win_state"] != "ok")
    if wbad:
        print("못 쟀음: " + " · ".join(f"{k} {v}세션" for k, v in wbad.items()))
    if wok:
        wn = sum(r["win_n"] for r in wok)
        we = sum(r["win_empty"] for r in wok)
        wb = sum(r["win_bad_time"] for r in wok)
        hi = sum(r["win_hi"] for r in wok)
        hr = sum(r["win_hi_radford"] for r in wok)
        print(f"창 {wn}개({len(wok)}세션) · 텍스트가 비어 못 잰 창 {we}개 · "
              f"시간 파싱 실패 낱말 {wb}개")
        print(f"문턱 {COMPRESSION_ALERT} 초과(잠정): {hi}창 "
              f"({hi / wn:.2%}, 분모 {wn})  · "
              f"원 문헌 {COMPRESSION_RADFORD} 초과: {hr}창 "
              f"({hr / wn:.2%}, 분모 {wn})")
        print(f"전체 창 압축률 최대: {max(r['win_max'] for r in wok):.2f}")
        flagged = [r for r in wok if r["win_hi"]]
        print(f"초과 창이 있는 세션: {len(flagged)}개"
              + ("  ← 선별이 아니라 우선순위. 사람이 들어야 확정된다"
                 if flagged else ""))
        for r in sorted(flagged, key=lambda x: -x["win_hi"])[:10]:
            print(f"    {r['name'][:44]}  {r['win_hi']}/{r['win_n']}창 "
                  f"· 최대 {r['win_max']:.2f}")

        # 문턱을 넘었는지와 **무관하게** 압축률 순으로 세운다. 이게 이 지표를
        # 쓰는 올바른 방법이다 — 문턱 2.4 의 재현율은 4.5% 라 넘은 것만 보면
        # 대부분을 놓친다. 넘은 창이 하나도 없어도 이 목록은 나온다.
        # 다만 세션마다 상위 3창만 모아 세우므로 **한 세션의 4위 창은 여기
        # 안 나온다.** 한 세션에 몰린 것까지 보려면 --csv 의 win_top_str 을
        # 쓰거나 그 세션만 따로 돌린다.
        order = sorted(((c, t, r["name"]) for r in wok
                        for t, c in r["win_top"]), key=lambda x: -x[0])[:15]
        print("\n  들을 순서 (압축률 높은 창부터 · 창 시작 시각):")
        for c, t, nm in order:
            mark = " ←문턱↑" if c > COMPRESSION_ALERT else ""
            print(f"    {c:5.2f}  {nm[:40]:40s} {hms(t)}"
                  f"~{hms(t + WINDOW_SEC)}{mark}")

    if a.csv:
        # win_top 은 리스트라 CSV 에 그대로 못 쓴다. 문자열 칸(win_top_str)만
        # 남기고 원본 리스트는 뺀다 — 뺀 것은 여기 한 칸뿐이다.
        flat = [{k: v for k, v in r.items() if k != "win_top"} for r in rows]
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
            wr.writeheader()
            wr.writerows(flat)
        print(f"saved {a.csv}")


if __name__ == "__main__":
    main()
