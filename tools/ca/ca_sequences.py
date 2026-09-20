# -*- coding: utf-8 -*-
"""CA 시퀀스 컬렉션 — 턴이 아니라 시퀀스 단위로 자르고, 다음 턴이 앞 턴을
어떻게 다루는지를 기술한다.

CA의 작업 방식을 따른다:
  · 한 현상의 사례를 전수 수집하고 (컬렉션)
  · 각 사례를 시퀀스 시작부터 닫히는 지점까지 통째로 배열하며
  · 각 턴이 '앞 턴을 어떻게 다뤘는지'를 관찰 가능한 사실로만 적고
  · 패턴에 어긋나는 일탈 사례를 따로 모은다.

기계는 읽을 자료를 만들 뿐, 무엇이 무슨 행위인지는 사람이 판정한다.
집계·비율은 내지 않는다.

세 컬렉션 (7/28 AICA 회의에서 교수님이 지적한 지점)
  A. 학생의 무지 주장 뒤 튜터의 처리   — "한없이 질문으로 간다"
  B. 학생이 튜터 진단에 불일치를 표시   — 튜터가 진단을 철회하는가
  C. 주제 전환이 개시되는 지점          — "20분이면 결판나야 한다"

출력: ~/Downloads/01_게일봇연구/gailbot_analysis/sequences/
"""
import csv
import re
from pathlib import Path

DL = Path.home() / "Downloads"
OUT = DL / "gailbot_analysis" / "sequences"
SOURCES = [
    ("cb", DL / "gailbot_results_v2"),
    ("gag", DL / "gailbot_results_ai_v2"),
    ("gag", DL / "gailbot_results_ai"),
]

_MARK = re.compile(r"\((\d[\d.]*)\)|\(\.\)|[∆∇]")
_PAUSE = re.compile(r"\((\d[\d.]*)\)")

# ── 참여자 자신이 드러내는 표시 (실제 코퍼스 출현형에서 뽑음) ──────────
NO_KNOW = re.compile(
    r"기억이?\s*(잘\s*)?안\s*나|기억이\s*잘\s*안|잘\s*모르|모르겠|"
    r"생각이\s*안\s*나|안\s*떠오르|떠오르지\s*않|없는\s*것\s*같|딱히\s*없")
DISAGREE = re.compile(r"^(아니(요|야|에요)?)\b|그건\s*아니|아닌\s*것\s*같|"
                      r"(같지|맞지)\s*않|힘들\s*것\s*같")
SHIFT = re.compile(r"두\s*번째|세\s*번째|네\s*번째|그럼\s*(이제\s*)?다음|"
                   r"다음\s*(문항|부분|거)|넘어가|다른\s*(부분|각도|거)")
QUESTION = re.compile(r"(까요|나요|은가요|는가요|어요\?|세요|을까|나\?|"
                      r"있으신가요|있나요|어떤가요|뭔가요|주세요)\s*$")
GIVE = re.compile(r"예를\s*들어|이렇게\s*(써|바꿔|고쳐)|이런\s*식으로|"
                  r"첫\s*문장|추천|~?로\s*묶|정리하면|이라고\s*쓰")

TUTOR_CUES = ["세션 끝내기", "수정이 완료되면", "잘 받았어요", "읽어봤는데",
              "좋은 질문", "인상적이에요", "해줄게", "말해줘", "올려주신 글"]
EXP_CUES = ["실험", "녹화", "나오시면 돼요", "안 들리시거나", "마이크"]


class Line:
    def __init__(self, n, sp, text, start, end):
        self.n, self.sp, self.text, self.start, self.end = n, sp, text, start, end
        self.silence = not sp

    @property
    def clean(self):
        return re.sub(r"\s{2,}", " ", _MARK.sub(" ", self.text)).strip()

    @property
    def dur(self):
        m = _PAUSE.search(self.text)
        return float(m.group(1)) if m and self.silence else 0.0


def load(folder):
    out, n = [], 0
    with open(folder / "conversation_turns.csv", newline="",
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            n += 1
            out.append(Line(n, (r.get("SPEAKER LABEL") or "").strip(),
                            r["TEXT"] or "", float(r["START TIME"]),
                            float(r["END TIME"])))
    return out


def roles(lines):
    sps = sorted({l.sp for l in lines if l.sp})
    if not sps:
        return {}
    sc = {}
    for sp in sps:
        ts = [l.clean for l in lines if l.sp == sp]
        sc[sp] = (sum(1 for c in TUTOR_CUES for t in ts if c in t),
                  sum(1 for c in EXP_CUES for t in ts if c in t),
                  sum(len(t.split()) for t in ts))
    tutor = max(sps, key=lambda s: (sc[s][0], sc[s][2]))
    out, rest = {tutor: "튜터"}, [s for s in sps if s != tutor]
    if len(rest) == 1:
        out[rest[0]] = "학생"
    elif rest:
        exp = max(rest, key=lambda s: sc[s][1])
        for s in rest:
            out[s] = "진행자" if (s == exp and sc[s][1] >= 2) else "학생"
    return out


def turns_only(lines):
    return [l for l in lines if not l.silence]


def describe(prev, cur, rl):
    """이 턴이 앞 턴을 어떻게 다뤘는지 — 관찰 가능한 사실만."""
    if prev is None:
        return ""
    who = rl.get(cur.sp, cur.sp)
    bits = []
    if cur.sp == prev.sp:
        bits.append("같은 화자 계속")
    if QUESTION.search(cur.clean[-12:]):
        bits.append("질문으로 종결")
    if GIVE.search(cur.clean):
        bits.append("예시·제안 제시")
    if SHIFT.search(cur.clean):
        bits.append("주제 전환 표지")
    if NO_KNOW.search(cur.clean):
        bits.append("무지 표시")
    if DISAGREE.search(cur.clean):
        bits.append("불일치 표시")
    # 앞 턴의 내용어를 얼마나 다시 쓰는지 = 같은 것을 계속 다루는지
    a = {w for w in prev.clean.split() if len(w) > 1}
    b = {w for w in cur.clean.split() if len(w) > 1}
    if a and len(a & b) / len(a) >= 0.3:
        bits.append("앞 턴 어휘 재사용")
    return f"[{who}] " + ", ".join(bits) if bits else f"[{who}]"


def fmt_time(s):
    return f"{int(s//60):02d}:{int(s%60):02d}"


def wrap(t, w=62, ind=" " * 12):
    ws, cur, out = t.split(), "", []
    for x in ws:
        if cur and len(cur) + len(x) + 1 > w:
            out.append(cur)
            cur = x
        else:
            cur = f"{cur} {x}".strip()
    out.append(cur)
    return ("\n" + ind).join(out)


def render(lines, rl, i0, i1, focal):
    rows = []
    for l in lines[i0:i1]:
        mark = "→" if l.n == focal else " "
        if l.silence:
            rows.append(f"{l.n:>4} {mark}          {l.text.strip()}")
        else:
            rows.append(f"{l.n:>4} {mark} {rl.get(l.sp, l.sp):<3}: "
                        f"{wrap(l.text.strip())}")
    return "\n".join(rows)


def sequence_bounds(lines, idx, back=1, fwd=6):
    """시퀀스 경계: 초점 앞 back턴부터, 뒤로는 주제 전환 표지가 나오거나
    fwd턴이 지날 때까지 (전환 표지 = 시퀀스가 닫히는 관찰 가능한 지점)."""
    i0 = idx
    cnt = 0
    while i0 > 0 and cnt < back:
        i0 -= 1
        if not lines[i0].silence:
            cnt += 1
    i1, cnt = idx + 1, 0
    while i1 < len(lines) and cnt < fwd:
        if not lines[i1].silence:
            cnt += 1
            if cnt > 1 and SHIFT.search(lines[i1].clean):
                i1 += 1
                break
        i1 += 1
    return i0, i1


# ── 세 컬렉션 정의 ────────────────────────────────────────────────────────
def coll_noknow(lines, rl):
    """A. 학생의 무지 주장."""
    out = []
    for i, l in enumerate(lines):
        if l.silence or rl.get(l.sp) != "학생":
            continue
        if NO_KNOW.search(l.clean):
            out.append(i)
    return out


def coll_disagree(lines, rl):
    """B. 학생의 불일치 표시."""
    out = []
    for i, l in enumerate(lines):
        if l.silence or rl.get(l.sp) != "학생":
            continue
        if DISAGREE.search(l.clean):
            out.append(i)
    return out


def coll_shift(lines, rl):
    """C. 주제 전환 개시 — 누가 개시했는지가 관건이라 화자 제한 없음."""
    out = []
    for i, l in enumerate(lines):
        if l.silence or not SHIFT.search(l.clean):
            continue
        out.append(i)
    return out


COLLECTIONS = [
    ("A_무지주장_뒤_튜터처리", coll_noknow,
     "학생이 “기억이 안 나”·“잘 모르겠어” 등으로 무지를 주장한 지점.\n"
     "읽을 것: 바로 다음 튜터 턴이 그것을 어떻게 다루는가 — 같은 질문을\n"
     "다시 묻는가, 질문을 바꾸는가, 답을 제공하고 넘어가는가. 그리고 그\n"
     "다음 학생 턴이 그 처리를 어떻게 받는가(3rd position)."),
    ("B_불일치_뒤_튜터처리", coll_disagree,
     "학생이 “아니”·“그건 아닌 것 같아” 등으로 튜터의 진단·제안에\n"
     "불일치를 표시한 지점.\n"
     "읽을 것: 튜터가 자기 진단을 유지하는가 철회하는가, 철회한다면\n"
     "무엇을 근거로 삼는가."),
    ("C_주제전환_개시", coll_shift,
     "“두 번째”·“다음”·“넘어가” 등 주제 전환이 개시된 지점.\n"
     "읽을 것: 전환을 개시한 쪽이 튜터인가 학생인가. 앞 주제가 어떤\n"
     "상태에서 닫혔는가(해결/미해결/방치)."),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sessions, seen = [], set()
    for scen, base in SOURCES:
        if not base.exists():
            continue
        for f in sorted(p for p in base.iterdir() if p.is_dir()):
            if f.name in seen or not (f / "conversation_turns.csv").exists():
                continue
            seen.add(f.name)
            ls = load(f)
            sessions.append((scen, f.name, ls, roles(ls)))

    index = ["# CA 시퀀스 컬렉션\n",
             "턴이 아니라 **시퀀스** 단위로 잘랐다. 각 사례 아래의 한 줄은\n"
             "*앞 턴을 어떻게 다뤘는지*를 관찰 가능한 사실로만 적은 것이며,\n"
             "무슨 행위인지의 판정은 읽는 사람이 한다.\n",
             "표기: `(n.n)` 침묵(초) · `(.)` 미세휴지 · `=` 래칭 · "
             "`낱말-` 절단 · → 초점 턴\n",
             "| 컬렉션 | 사례 수 | 세션 수 |", "|---|---|---|"]

    for key, detect, guide in COLLECTIONS:
        body = [f"# {key.split('_', 1)[1].replace('_', ' ')}\n", guide, ""]
        total = nsess = 0
        for scen, name, lines, rl in sessions:
            hits = detect(lines, rl)
            if not hits:
                continue
            nsess += 1
            total += len(hits)
            body.append(f"\n---\n\n## [{scen}] {name}\n")
            for idx in hits:
                l = lines[idx]
                i0, i1 = sequence_bounds(lines, idx)
                body.append(f"\n**{fmt_time(l.start)}** (행 {l.n})\n")
                body.append("```")
                body.append(render(lines, rl, i0, i1, l.n))
                body.append("```")
                # 초점 이후 각 턴이 앞 턴을 어떻게 다뤘는지
                seq = [x for x in lines[idx:i1] if not x.silence]
                notes = []
                for a, b in zip(seq, seq[1:]):
                    notes.append(f"- {b.n}행 {describe(a, b, rl)}")
                if notes:
                    body.append("\n다음 턴들이 앞 턴을 다룬 방식:\n"
                                + "\n".join(notes))
        (OUT / f"{key}.md").write_text("\n".join(body), encoding="utf-8")
        index.append(f"| {key} | {total} | {nsess} |")
        print(f"{key}: {total}건 / {nsess}개 세션")

    (OUT / "INDEX.md").write_text("\n".join(index), encoding="utf-8")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
