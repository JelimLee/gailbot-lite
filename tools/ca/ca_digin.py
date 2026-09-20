# -*- coding: utf-8 -*-
"""Dig-in 시퀀스 컬렉션 — 튜터가 구체적 경험·사례를 캐물어 들어가는 연쇄와
그 귀결을 시퀀스 단위로 모은다.

배경(7/28 AICA 회의): 튜터는 자소서에 넣을 구체적 사례를 찾으려 한 주제를
계속 파고드는데, 학생은 이를 회피하거나 거부한다(교수님: "한 주제로 계속
빠져나간다", 연구팀원: "글로 안 들어가는 느낌"). 그렇다면 **파고들기가
성사된 사례는 어떤 모습인가**를 보려는 것이 이 컬렉션의 목적이다.

시퀀스 경계
  개시: 튜터가 구체적 사례·경험을 요청하는 턴 (구체적으로/예를 들어/
        어떤 상황/기억나세요 + 질문 종결)
  종료: 학생이 글에 넣을 후보 문장을 내놓거나(성사), 주제가 전환되거나,
        회피·거부가 나오고 더 이상 같은 것을 다루지 않을 때(불발)

귀결 판정은 **참여자가 스스로 드러낸 것**만 근거로 삼는다.
  성사 신호 : 학생이 후보 문장을 제시("이렇게 쓰면 되나", "라고 쓰면"),
              또는 수정을 완료했다고 보고("넣었어", "고쳤어")
  불발 신호 : 무지 주장, 글자수·중복을 이유로 한 거부, 어렵다는 호소,
              후보 문장 없이 튜터가 주제를 전환
어느 쪽 신호도 없으면 '미결'로 두고 사람이 읽도록 남긴다.

출력: ~/Downloads/01_게일봇연구/gailbot_analysis/sequences/D_digin_*.md
"""
import csv
import re
from pathlib import Path

DL = Path.home() / "Downloads"
OUT = DL / "gailbot_analysis" / "sequences"
SOURCES = [("cb", DL / "gailbot_results_v2"),
           ("gag", DL / "gailbot_results_ai_v2"),
           ("gag", DL / "gailbot_results_ai")]

_MARK = re.compile(r"\((\d[\d.]*)\)|\(\.\)|[∆∇]")
_PAUSE = re.compile(r"\((\d[\d.]*)\)")

# 튜터의 파고들기 개시
DIGIN = re.compile(
    r"구체적으로|구체적인 (사례|경험|상황|예)|예를 들어 (어떤|무슨)|"
    r"어떤 (상황|경험|일|순간|계기)|기억(나|에 남)|있으셨나요|있었나요|"
    r"어떻게 (하셨|했)")
QEND = re.compile(r"(까요|나요|은가요|는가요|세요|을까|가요|어요)\s*[?]?\s*$")

# 학생이 드러내는 귀결 신호
CAND = re.compile(r"이렇게 쓰[면가]|이렇게 하면 [되될]|라고 쓰면|"
                  r"이런 식으로 쓰|이렇게 적|이렇게 바꿔|썼는데 어때")
DONE = re.compile(r"수정했|고쳤|바꿨|넣었|적었|썼어")
REFUSE = re.compile(r"글자[수소]\s*제한|글자가|중복|이미 (썼|말했|적었)|"
                    r"쓸 수 없|넣고 싶지|쓰고 싶지")
TROUBLE = re.compile(r"기억이?\s*(잘\s*)?안\s*나|잘\s*모르|모르겠|어렵|"
                     r"못\s*(적|쓰)겠|힘들 것 같|없는 것 같|딱히 없")
SHIFT = re.compile(r"두\s*번째|세\s*번째|그럼\s*(이제\s*)?다음|다음\s*(문항|부분|거)|"
                   r"넘어가|다른\s*(부분|각도|거)")

TUTOR_CUES = ["세션 끝내기", "수정이 완료되면", "잘 받았어요", "읽어봤는데",
              "좋은 질문", "인상적이에요", "해줄게", "말해줘", "올려주신 글"]
EXP_CUES = ["실험", "녹화", "나오시면 돼요", "안 들리시거나", "마이크"]


class Line:
    def __init__(self, n, sp, text, start):
        self.n, self.sp, self.text, self.start = n, sp, text, start
        self.silence = not sp

    @property
    def clean(self):
        return re.sub(r"\s{2,}", " ", _MARK.sub(" ", self.text)).strip()

    @property
    def sil(self):
        m = _PAUSE.search(self.text)
        return float(m.group(1)) if m and self.silence else 0.0


def load(folder):
    out = []
    with open(folder / "conversation_turns.csv", newline="",
              encoding="utf-8") as f:
        for i, r in enumerate(csv.DictReader(f), 1):
            out.append(Line(i, (r.get("SPEAKER LABEL") or "").strip(),
                            r["TEXT"] or "", float(r["START TIME"])))
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
    else:
        exp = max(rest, key=lambda s: sc[s][1]) if rest else None
        for s in rest:
            out[s] = "진행자" if (s == exp and sc[s][1] >= 2) else "학생"
    return out


def fmt(s):
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
        m = "→" if l.n == focal else " "
        if l.silence:
            rows.append(f"{l.n:>4} {m}          {l.text.strip()}")
        else:
            rows.append(f"{l.n:>4} {m} {rl.get(l.sp, l.sp):<3}: "
                        f"{wrap(l.text.strip())}")
    return "\n".join(rows)


def trace(lines, rl, start_idx, max_turns=10):
    """개시 턴부터 귀결이 드러날 때까지 추적.
    반환: (끝 인덱스, 귀결, 근거 행번호, 근거 표현)"""
    seen = 0
    for j in range(start_idx + 1, len(lines)):
        l = lines[j]
        if l.silence:
            continue
        seen += 1
        who = rl.get(l.sp)
        if who == "학생":
            for rx, label in ((CAND, "성사: 후보 문장 제시"),
                              (DONE, "성사: 수정 완료 보고")):
                m = rx.search(l.clean)
                if m:
                    return j + 1, label, l.n, m.group(0)
            for rx, label in ((REFUSE, "불발: 거부(글자수·중복 등)"),
                              (TROUBLE, "불발: 곤란·무지 표시")):
                m = rx.search(l.clean)
                if m:
                    return j + 1, label, l.n, m.group(0)
        if who == "튜터" and seen > 1 and SHIFT.search(l.clean):
            m = SHIFT.search(l.clean)
            return j + 1, "불발: 후보 없이 튜터가 주제 전환", l.n, m.group(0)
        if seen >= max_turns:
            return j + 1, "미결", 0, ""
    return len(lines), "미결", 0, ""


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

    buckets = {"성사": [], "불발": [], "미결": []}
    for scen, name, lines, rl in sessions:
        for i, l in enumerate(lines):
            if l.silence or rl.get(l.sp) != "튜터":
                continue
            if not (DIGIN.search(l.clean) and QEND.search(l.clean[-14:])):
                continue
            end, verdict, ev_n, ev_t = trace(lines, rl, i)
            block = [f"\n**{fmt(l.start)}** [{scen}] {name} (개시 {l.n}행)"
                     f" — {verdict}"
                     + (f" · {ev_n}행 “{ev_t}”" if ev_n else "") + "\n",
                     "```", render(lines, rl, i, end, l.n), "```"]
            buckets[verdict.split(":")[0]].append("\n".join(block))

    head = ("# Dig-in 시퀀스 — 튜터가 구체적 사례를 캐물어 들어간 연쇄\n\n"
            "튜터가 구체적 경험·사례를 요청하는 턴에서 시작해, 학생이 글에\n"
            "넣을 후보 문장을 내놓거나(성사) 회피·거부하거나 주제가 전환될\n"
            "때까지(불발)를 잘랐다. 귀결 판정은 참여자가 스스로 드러낸 표현만\n"
            "근거로 삼았고, 어느 신호도 없으면 미결로 두었다.\n\n"
            "표기: `(n.n)` 침묵(초) · `(.)` 미세휴지 · `=` 래칭 · → 개시 턴\n")

    for k, items in buckets.items():
        (OUT / f"D_digin_{k}.md").write_text(
            head + f"\n## {k} — {len(items)}건\n" + "\n".join(items),
            encoding="utf-8")
        print(f"{k}: {len(items)}건")
    print(f"→ {OUT}")


if __name__ == "__main__":
    main()
