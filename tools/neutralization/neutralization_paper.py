# -*- coding: utf-8 -*-
"""논문용 중립화 심층 분석 — 기존 컬렉션(사례 수집) 위에 얹는 4가지.

  A. 추궁 연쇄(pursuit chain): 입장요청 → [중립화 → 저항]* → 귀결.
     연쇄 단위로 잘라야 "저항이 상승해도 중립이 유지되는가"가 보인다.
  B. 양보(concession) 형식: GPT가 결국 입장을 낼 때의 완화 표지.
  C. 비선호 타이밍: 중립화 턴 앞 갭·턴 길이 vs 일반 GPT 턴 (게일봇 시간값).
  D. 직답 대조(일탈 사례): 즉시 입장을 받은 입장요청 — 중립화의 선택성.

자료: 2025 수작업 CA 전사본(9) + 2025 게일봇(18) + 2026 게일봇(10). 출처 표기.
출력: ~/Downloads/01_게일봇연구/고베 학회 준비/논문분석_*.md
"""
import csv
import re
import statistics as st
from pathlib import Path

from neutralization_docx import (from_doc, from_gailbot,
                                 neutral_marks, RESIST, OPINION)
from kobe_sequences import SOLICIT

DL = Path.home() / "Downloads"
OUT = DL / "고베 학회 준비"
CA2025 = DL / "0 ChatGPT4o 전사전환 완료"
GB2025 = DL / "gailbot_results_gpt2025"
GB2026 = DL / "gailbot_results_gpt2026"

# 양비론의 영어 관용 표현 — BOTHSIDES가 놓친 것들
BOTHSIDES_EXTRA = re.compile(
    r"double[- ]edged sword|moderation is key|a hybrid|sweet spot|"
    r"depends on the|for some .{0,30}(but|while|others)|"
    r"there('s| is) (a )?balance|both .{0,20}(and|as well)|"
    r"양날의 검|절충|균형이 (중요|필요)|사람마다|경우에 따라", re.I)
# 방향성 있는 입장 — 한쪽을 실제로 고른 표현만
POLARITY = re.compile(
    r"\b(positive|negative|better|worse|best|worst|should|shouldn'?t|"
    r"agree|disagree|in favou?r|against|more important|outweigh)\b|"
    r"긍정적|부정적|더 (나|좋|현명|중요)|찬성|반대|해야 (한다|해)|"
    r"하지 말아야", re.I)

# 1인칭 입장 틀 — 이것만으로는 입장이 아니고, POLARITY와 함께여야 한다
STANCE_FRAME = re.compile(
    r"in my opinion|personally|i(\'d| would)? (say|think|believe|feel|pick|"
    r"choose|go with|prefer)|my (choice|pick|opinion|view|take) (is|would)|"
    r"if i had to|나라면|저라면|개인적으로|제 생각(에는|엔)|내 생각(에는|엔)|"
    r"고르자면|추천(할게|드리|합니다|해요)", re.I)

# 양보 발화 안의 완화(헤지) 표지
HEDGE = re.compile(
    r"아마도|것 같(아|아요|습니다)|probably|perhaps|maybe|i think|"
    r"slightly|약간|일단|우선은|굳이|하지만|그래도|다만", re.I)


def wrap(t, w=66, ind=" " * 11):
    ws, cur, out = t.split(), "", []
    for x in ws:
        if cur and len(cur) + len(x) + 1 > w:
            out.append(cur)
            cur = x
        else:
            cur = f"{cur} {x}".strip()
    out.append(cur)
    return ("\n" + ind).join(out)


def render(turns, i0, i1, marks=None):
    rows = []
    for t in turns[i0:i1]:
        m = "→" if marks and t.n in marks else " "
        rows.append(f"{t.n:>4} {m} {t.sp or '·'}: {wrap(t.text[:400])}")
    return "\n".join(rows)


def load_sessions():
    out = []
    for p in sorted(CA2025.glob("*.doc*")):
        out.append(("2025", "수작업CA", p.stem, from_doc(p)))
    manual = {re.sub(r"[^가-힣A-Za-z]", "", p.stem)[:6]
              for p in CA2025.glob("*.doc*")}
    for base, year in ((GB2025, "2025"), (GB2026, "2026")):
        if not base.exists():
            continue
        for f in sorted(x for x in base.iterdir() if x.is_dir()):
            if not (f / "conversation_turns.csv").exists():
                continue
            out.append((year, "게일봇", f.name, from_gailbot(f)))
    return out


def classify_g(text):
    marks = neutral_marks(text)
    if BOTHSIDES_EXTRA.search(text) and "양비론" not in marks:
        marks = marks + ["양비론"]
    if len(marks) >= 2:
        return "중립화", marks
    # 양비론 표지가 하나라도 있으면 입장표명으로 보지 않는다.
    # "I'd say it's a double-edged sword"는 1인칭 틀만 입장이고 내용은 유보다.
    if marks:
        return "중립화", marks
    # 입장은 1인칭 틀(I think)만으로 성립하지 않는다. 방향을 골라야 한다.
    if STANCE_FRAME.search(text) and POLARITY.search(text):
        return "입장표명", []
    return "기타", []


def classify_u(text):
    for label, rx in RESIST.items():
        m = rx.search(text)
        if m:
            return "저항", f"{label} “{m.group(0)}”"
    if SOLICIT.search(text):
        return "재요청", ""
    if OPINION.search(text):
        return "정렬", ""
    return "기타", ""


def build_chains(turns):
    """입장요청에서 시작해 귀결까지 추적."""
    chains, used = [], set()
    for i, t in enumerate(turns):
        if t.sp != "U" or i in used or not SOLICIT.search(t.text):
            continue
        chain = {"start": i, "rounds": 0, "events": [(i, "U", "입장요청", "")],
                 "outcome": "미결", "end": i}
        j = i
        pursued = False
        while j < len(turns) - 1:
            # 다음 G 턴
            g = next((k for k in range(j + 1, min(j + 3, len(turns)))
                      if turns[k].sp == "G"), None)
            if g is None:
                break
            gk, marks = classify_g(turns[g].text)
            chain["events"].append((g, "G", gk, " + ".join(marks)))
            chain["end"] = g
            if gk == "입장표명":
                chain["outcome"] = "양보" if pursued else "직답"
                break
            if gk != "중립화":
                # 기타 응답이라도 사용자가 계속 추궁하면 연쇄를 이어간다
                u2 = next((k for k in range(g + 1, min(g + 3, len(turns)))
                           if turns[k].sp == "U"), None)
                if u2 is not None:
                    uk2, ev2 = classify_u(turns[u2].text)
                    if uk2 in ("저항", "재요청"):
                        chain["events"].append((u2, "U", uk2, ev2))
                        chain["end"] = u2
                        used.add(u2)
                        pursued = True
                        chain["rounds"] += 1
                        j = u2
                        continue
                chain["outcome"] = "기타응답"
                break
            # 중립화 → 다음 U 턴이 추궁인가
            u = next((k for k in range(g + 1, min(g + 3, len(turns)))
                      if turns[k].sp == "U"), None)
            if u is None:
                chain["outcome"] = "중립유지(종료)"
                break
            uk, ev = classify_u(turns[u].text)
            chain["events"].append((u, "U", uk, ev))
            chain["end"] = u
            used.add(u)
            if uk in ("저항", "재요청"):
                pursued = True
                chain["rounds"] += 1
                j = u
                continue
            chain["outcome"] = "정렬" if uk == "정렬" else "이탈"
            break
        if len(chain["events"]) >= 2:
            chains.append(chain)
    return chains


# ── C. 게일봇 시간값: 중립화 턴의 선행 갭·길이 ─────────────────────────────
def timing_rows(base):
    rows = []
    for f in sorted(x for x in base.iterdir() if x.is_dir()):
        p = f / "conversation_turns.csv"
        if not p.exists():
            continue
        rs = [((r["SPEAKER LABEL"] or "").strip(), (r["TEXT"] or "").strip(),
               float(r["START TIME"]), float(r["END TIME"]))
              for r in csv.DictReader(open(p, encoding="utf-8"))]
        turns = from_gailbot(f)
        # from_gailbot은 화자와 텍스트가 모두 있는 행만 남긴다.
        # 침묵 행(화자 없음, 텍스트 "(2.1)")을 똑같이 걸러야 정렬이 맞는다.
        nonempty = [r for r in rs if r[0] and r[1]]
        for t, r in zip(turns, nonempty):
            t.start, t.end = r[2], r[3]
        for k, t in enumerate(turns):
            gap = t.start - turns[k - 1].end if k else None
            if t.sp != "G" or gap is None or gap < 0 or gap > 120:
                continue
            kind, _ = classify_g(t.text)
            rows.append((kind, gap, t.end - t.start, len(t.text)))
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    sessions = load_sessions()

    # ── A + B + D ──
    all_chains, concessions, directs = [], [], []
    for year, src, name, turns in sessions:
        for c in build_chains(turns):
            c.update(year=year, src=src, name=name, turns=turns)
            all_chains.append(c)
            last = c["events"][-1]
            if c["outcome"] == "양보":
                concessions.append(c)
            elif c["outcome"] == "직답":
                directs.append(c)

    # A. 추궁 연쇄
    md = ["# 논문분석 A — 추궁 연쇄(pursuit chain)\n",
          "입장요청에서 시작해 귀결(양보/정렬/중립유지/이탈)까지를 한 단위로 "
          "잘랐다. 저항이 몇 라운드 이어지는지, 무엇으로 끝나는지가 "
          "낱개 사례로는 보이지 않기 때문이다.\n"]
    from collections import Counter
    by_year = {}
    for c in all_chains:
        by_year.setdefault(c["year"], []).append(c)
    md.append("| 연도 | 연쇄 수 | 직답 | 양보 | 정렬 | 중립유지 | 기타 | 평균 추궁 라운드 |")
    md.append("|---|---|---|---|---|---|---|---|")
    for y, cs in sorted(by_year.items()):
        oc = Counter(c["outcome"] for c in cs)
        rounds = [c["rounds"] for c in cs if c["rounds"] > 0]
        md.append(f"| {y} | {len(cs)} | {oc.get('직답',0)} | {oc.get('양보',0)} | "
                  f"{oc.get('정렬',0)} | {oc.get('중립유지(종료)',0)} | "
                  f"{oc.get('기타응답',0)+oc.get('이탈',0)+oc.get('미결',0)} | "
                  f"{st.mean(rounds):.1f}" if rounds else
                  f"| {y} | {len(cs)} | {oc.get('직답',0)} | {oc.get('양보',0)} | "
                  f"{oc.get('정렬',0)} | {oc.get('중립유지(종료)',0)} | "
                  f"{oc.get('기타응답',0)+oc.get('이탈',0)+oc.get('미결',0)} | –")
        md[-1] += " |"
    md.append("\n## 추궁이 2라운드 이상 이어진 연쇄 (전량)\n")
    for c in sorted(all_chains, key=lambda x: -x["rounds"]):
        if c["rounds"] < 2:
            continue
        md.append(f"\n### [{c['name']} · {c['year']} {c['src']}] "
                  f"추궁 {c['rounds']}라운드 → **{c['outcome']}**\n")
        lo = max(0, c["start"] - 0)
        hi = min(len(c["turns"]), c["end"] + 2)
        md.append("```")
        md.append(render(c["turns"], lo, hi,
                         {c["turns"][e[0]].n for e in c["events"]}))
        md.append("```")
        md.append("\n연쇄 구조: " + " → ".join(
            f"{e[2]}" + (f"({e[3]})" if e[3] else "") for e in c["events"]))
    (OUT / "논문분석_A_추궁연쇄.md").write_text("\n".join(md), encoding="utf-8")

    # B. 양보 형식
    md = ["# 논문분석 B — 양보(concession)의 형식\n",
          "추궁 끝에 GPT가 입장을 낼 때, 그 발화가 어떻게 완화되는지. "
          "양보조차 헤지된다면 중립화가 국소적 회피가 아니라 일관된 "
          "산출 형식임을 보여준다.\n",
          f"양보 {len(concessions)}건 / 직답 {len(directs)}건\n"]
    for c in concessions:
        gi = next(e[0] for e in reversed(c["events"]) if e[1] == "G")
        text = c["turns"][gi].text
        hedges = HEDGE.findall(text)
        md.append(f"\n### [{c['name']} · {c['year']} {c['src']}] "
                  f"추궁 {c['rounds']}라운드 뒤 양보 — "
                  f"완화 표지 {len(hedges)}개: "
                  f"{', '.join(dict.fromkeys(hedges)) or '없음'}\n")
        # 무엇이 추궁이고 무엇이 양보인지 보이도록 연쇄 전체를 싣는다
        md.append("```")
        md.append(render(c["turns"], c["start"],
                         min(len(c["turns"]), gi + 1),
                         {c["turns"][e[0]].n for e in c["events"]}))
        md.append("```")
        md.append("\n연쇄 구조: " + " → ".join(
            f"{e[2]}" + (f"({e[3]})" if e[3] else "") for e in c["events"]))
        md.append(f"\n**양보 턴({c['turns'][gi].n}행)의 완화 표지**: "
                  + (", ".join(f"`{h}`" for h in dict.fromkeys(hedges))
                     if hedges else "없음 — 완화 없이 입장을 냄"))
    (OUT / "논문분석_B_양보형식.md").write_text("\n".join(md), encoding="utf-8")

    # C. 비선호 타이밍 (게일봇만)
    md = ["# 논문분석 C — 비선호 형식의 타이밍 (게일봇 시간값)\n",
          "중립화 턴의 선행 갭과 턴 길이를 GPT 턴 유형별로 비교했다.\n",
          "> **타이밍을 비선호의 증거로 읽지 말 것.** Kendrick & Torreira 2015 는 "
          "195건에서 선호/비선호의 응답 시간이 체계적으로 다르지 않았고, 차이는 "
          "700ms 이상에서만 뚜렷했다고 보고했다. 비선호의 증거는 **턴 형식**이고, "
          "여기 표는 그 형식에 딸린 부수 기술로만 쓴다.\n"]
    for base, y in ((GB2025, "2025"), (GB2026, "2026")):
        if not base.exists():
            continue
        rows = timing_rows(base)
        md.append(f"\n## {y} (게일봇 {len(rows)}개 GPT 턴)\n")
        md.append("| GPT 턴 유형 | n | 선행 갭 중앙 | 턴 길이 중앙 | 글자수 중앙 |")
        md.append("|---|---|---|---|---|")
        for kind in ("중립화", "입장표명", "기타"):
            v = [r for r in rows if r[0] == kind]
            if not v:
                continue
            md.append(f"| {kind} | {len(v)} | {st.median(x[1] for x in v):.1f}s | "
                      f"{st.median(x[2] for x in v):.1f}s | "
                      f"{st.median(x[3] for x in v):.0f}자 |")
    md.append("\n주의: 갭에는 시스템 지연이 포함되므로 절대값이 아니라 "
              "**유형 간 차이**만 해석할 것. 수작업 전사본의 (2.0) 표기는 "
              "별도 수동 확인 필요.")
    (OUT / "논문분석_C_비선호타이밍.md").write_text("\n".join(md), encoding="utf-8")

    # D. 직답 대조
    md = ["# 논문분석 D — 즉시 입장표명 (일탈 사례)\n",
          "**무엇을 모았나**: 사용자가 입장을 물었을 때 GPT가 추궁을 받기 전에 "
          "**곧바로 한쪽을 고른** 경우입니다. 대부분의 연쇄가 중립화로 가는데 "
          "이들만 다르므로, CA에서 말하는 **일탈 사례(deviant case)**에 해당합니다.\n",
          "**왜 필요한가**: 중립화가 무조건적 정책이라면 예외가 없어야 합니다. "
          "예외가 있다면 중립화는 **선택적**이며, 무엇이 그 선택을 가르는지"
          "(화제가 사적 취향인가 공적 쟁점인가, 논쟁적인가 아닌가)가 "
          "분석 대상이 됩니다.\n",
          "**판정 기준**: 양비론 표지(both/depends/double-edged sword/절충 등)가 "
          "하나도 없고, 1인칭 입장 틀(I think/personally/나라면)과 방향성 어휘"
          "(positive/should/better/찬성 등)가 함께 나타난 경우만.\n",
          f"즉시 입장표명 {len(directs)}건 (전체 연쇄 대비 소수)\n"]
    for c in directs[:20]:
        md.append(f"\n### [{c['name']} · {c['year']} {c['src']}]\n")
        md.append("```")
        md.append(render(c["turns"], c["start"],
                         min(len(c["turns"]), c["end"] + 1),
                         {c["turns"][e[0]].n for e in c["events"]}))
        md.append("```")
    (OUT / "논문분석_D_직답대조.md").write_text("\n".join(md), encoding="utf-8")

    print(f"세션 {len(sessions)}개 · 연쇄 {len(all_chains)}건")
    print(f"  양보 {len(concessions)} / 직답 {len(directs)} / "
          f"2라운드+ 추궁 {sum(1 for c in all_chains if c['rounds'] >= 2)}")
    print(f"→ {OUT}/논문분석_A~D_*.md")


if __name__ == "__main__":
    main()
