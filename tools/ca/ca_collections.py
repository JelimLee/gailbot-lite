# -*- coding: utf-8 -*-
"""CA 컬렉션 빌더: 순차분석을 위한 현상별 사례 수집 + 발췌 생성.

대화분석(CA)의 작업 방식(전수조사 → 컬렉션 → 말차례 추적 → 패턴 분류)에서
기계가 맡는 부분: 현상 후보를 전부 찾아 앞뒤 맥락과 함께 논문 형식의
발췌로 잘라 모은다. 해석은 사람이 한다.

수집하는 현상
  1. 빠른 발화권 이동 (끼어들기 후보) — 화자 교체 간격(FTO) ≤ 0.2초.
     ※ Whisper 모노 전사는 겹침말을 기록할 수 없으므로(코퍼스 전체
       overlap 0건) 이는 겹침의 '대리 지표'다. 타임스탬프로 원본
       음성을 확인해 실제 겹침 여부를 판정할 것.
  2. 재발화 — 같은 화자가 비슷한 발화를 반복 (STT 실패 후 수정 신호)
  3. 수정개시 — 턴 첫머리 "아니(요)", "그게 아니라"
  4. 확인요청·제안 — 튜터 턴이 "~할까요/맞습니까/어떨까요" 류로 종결
  5. 긴 갭 — 8초 이상 침묵 뒤 학생 턴 (단어 찾기/망설임/이탈 후보)
  6. 퍼즈 밀집 턴 — 학생 턴 안에 0.8초+ 퍼즈가 3회 이상

출력 (~/Downloads/01_게일봇연구/gailbot_analysis/collections/)
  INDEX.md            현상별·세션별 건수, 방법·한계 설명
  0N_<현상>.md        현상별 전체 사례 발췌 (논문 형식, ▶ = 초점 턴)
  maps/<세션>.md      세션별 궤적 지도 (사건을 시간순으로 나열)

데이터 소스: 파일별로 v2(재전사)가 완료돼 있으면 v2, 아니면 v1.
"""
import csv
import re
from pathlib import Path

from step0_speaker_check import TUTOR_CUES, EXPERIMENTER_CUES

DL = Path.home() / "Downloads"
SOURCES = [
    ("cb", DL / "gailbot_results", DL / "gailbot_results_v2"),
    ("gag", DL / "gailbot_results_ai", DL / "gailbot_results_ai_v2"),
]
OUT = DL / "gailbot_analysis" / "collections"

FTO_MAX = 0.2          # 빠른 발화권 이동 판정(초)
LONG_GAP = 8.0         # 긴 갭(초)
REPEAT_JACCARD = 0.55  # 재발화 유사도
PAUSE_DENSE_N = 3      # 퍼즈 밀집: 0.8초+ 퍼즈 최소 횟수

_MARK_RE = re.compile(r"\([\d.]+\)|\(\.\)|[∆∇]")
_PAUSE_RE = re.compile(r"\((\d[\d.]*)\)")
_REPAIR_RE = re.compile(r"^(아니요?|아니에요|아냐|아니야)\b|^.{0,8}그게 아니라")
_CONFIRM_RE = re.compile(
    r"(할까요|볼까요|맞습니까|맞나요|맞죠|맞으시죠|말씀이시죠|어떨까요|"
    r"어떤가요|괜찮을까요|해볼까요|볼게요)\s*$")


# ── 데이터 로딩 ────────────────────────────────────────────────────────────
class Line:
    """전사본의 한 줄: 발화 턴 또는 침묵."""
    def __init__(self, num, speaker, text, start, end):
        self.num, self.speaker, self.text = num, speaker, text
        self.start, self.end = start, end

    @property
    def is_silence(self):
        return not self.speaker

    @property
    def clean(self):
        return _MARK_RE.sub(" ", self.text).strip()

    @property
    def silence_dur(self):
        m = _PAUSE_RE.search(self.text)
        return float(m.group(1)) if m else 0.0


def resolve_folder(name, v1_base, v2_base):
    v2 = v2_base / name
    if (v2 / "utt.toml").exists():
        return v2, "v2"
    return v1_base / name, "v1"


def load_lines(folder):
    lines, num = [], 0
    with open(folder / "conversation_turns.csv", newline="",
              encoding="utf-8") as f:
        for row in csv.DictReader(f):
            num += 1
            lines.append(Line(num, (row.get("SPEAKER LABEL") or "").strip(),
                              row["TEXT"] or "",
                              float(row["START TIME"]),
                              float(row["END TIME"])))
    return lines


def guess_roles(lines):
    """화자 라벨 → 역할(튜터/학생/진행자) 추정."""
    speakers = sorted({l.speaker for l in lines if l.speaker})
    score = {}
    for sp in speakers:
        texts = [l.clean for l in lines if l.speaker == sp]
        cue = sum(w for ph, w in TUTOR_CUES for t in texts if ph in t)
        exp = sum(1 for ph in EXPERIMENTER_CUES for t in texts if ph in t)
        words = sum(len(t.split()) for t in texts)
        score[sp] = (cue, exp, words)
    if not speakers:
        return {}
    tutor = max(speakers, key=lambda s: (score[s][0], score[s][2]))
    roles = {tutor: "튜터"}
    rest = [s for s in speakers if s != tutor]
    if len(rest) == 1:
        roles[rest[0]] = "학생"
    else:
        exp_sp = max(rest, key=lambda s: score[s][1])
        if score[exp_sp][1] >= 2:
            roles[exp_sp] = "진행자"
            for s in rest:
                if s != exp_sp:
                    roles[s] = "학생"
        else:
            stu = max(rest, key=lambda s: score[s][2])
            roles[stu] = "학생"
            for s in rest:
                if s != stu:
                    roles[s] = f"기타({s})"
    return roles


# ── 현상 탐지기: (초점 줄번호, 비고) 목록을 돌려준다 ──────────────────────
def det_fast_transfer(lines, roles):
    out, prev = [], None
    for l in lines:
        if l.is_silence:
            continue
        if prev is not None and l.speaker != prev.speaker:
            fto = l.start - prev.end
            if 0 <= fto <= FTO_MAX:
                a = roles.get(prev.speaker, prev.speaker)
                b = roles.get(l.speaker, l.speaker)
                out.append((l.num, f"{a}→{b}, 간격 {fto:.2f}초"))
        prev = l
    return out


def det_repeat(lines, roles):
    turns = [l for l in lines if not l.is_silence]
    out = []
    for i, a in enumerate(turns):
        wa = set(a.clean.split())
        if len(wa) < 4:
            continue
        for b in turns[i + 1:i + 5]:
            if b.speaker != a.speaker:
                continue
            wb = set(b.clean.split())
            if len(wb) < 4:
                continue
            j = len(wa & wb) / len(wa | wb)
            if j >= REPEAT_JACCARD:
                out.append((b.num, f"{roles.get(a.speaker, a.speaker)}, "
                                   f"{a.num}행과 유사도 {j:.2f}"))
                break
    return out


def det_repair_init(lines, roles):
    return [(l.num, roles.get(l.speaker, l.speaker))
            for l in lines
            if not l.is_silence and _REPAIR_RE.search(l.clean)]


def det_confirm(lines, roles):
    return [(l.num, "")
            for l in lines
            if not l.is_silence and roles.get(l.speaker) == "튜터"
            and _CONFIRM_RE.search(l.clean[-25:])]


def det_long_gap(lines, roles):
    out = []
    for i, l in enumerate(lines):
        if l.is_silence and l.silence_dur >= LONG_GAP:
            nxt = next((x for x in lines[i + 1:] if not x.is_silence), None)
            if nxt is not None and roles.get(nxt.speaker) == "학생":
                out.append((l.num, f"{l.silence_dur:.1f}초 → 학생"))
    return out


def det_pause_dense(lines, roles):
    out = []
    for l in lines:
        if l.is_silence or roles.get(l.speaker) != "학생":
            continue
        n = sum(1 for m in _PAUSE_RE.finditer(l.text)
                if float(m.group(1)) >= 0.8)
        if n >= PAUSE_DENSE_N:
            out.append((l.num, f"0.8초+ 퍼즈 {n}회"))
    return out


PHENOMENA = [
    ("01_끼어들기후보", "빠른 발화권 이동 (겹침 후보, FTO ≤ 0.2초)",
     det_fast_transfer, 3, 4),
    ("02_재발화", "같은 화자의 유사 발화 반복", det_repeat, 4, 3),
    ("03_수정개시", "턴 첫머리 '아니(요)' / '그게 아니라'",
     det_repair_init, 3, 4),
    ("04_확인요청", "튜터의 확인요청·제안형 턴 종결", det_confirm, 2, 4),
    ("05_긴갭", f"{LONG_GAP:.0f}초+ 침묵 뒤 학생 턴", det_long_gap, 3, 3),
    ("06_퍼즈밀집턴", "학생 턴 내 0.8초+ 퍼즈 3회 이상",
     det_pause_dense, 2, 3),
]


# ── 발췌·지도 출력 ─────────────────────────────────────────────────────────
def fmt_time(sec):
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def wrap(text, width=64, indent=" " * 14):
    words, cur, out = text.split(), "", []
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            out.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    out.append(cur)
    return ("\n" + indent).join(out)


def render_excerpt(lines, roles, focal, before, after):
    i = next(idx for idx, l in enumerate(lines) if l.num == focal)
    lo, hi = max(0, i - before), min(len(lines), i + after + 1)
    rows = []
    for l in lines[lo:hi]:
        mark = "▶" if l.num == focal else " "
        if l.is_silence:
            rows.append(f"{l.num:>4}. {mark}       {l.text.strip()}")
        else:
            role = roles.get(l.speaker, l.speaker)
            rows.append(f"{l.num:>4}. {mark} {role:<3}: "
                        f"{wrap(l.text.strip())}")
    return "\n".join(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "maps").mkdir(exist_ok=True)

    sessions = []          # (scenario, name, ver, lines, roles)
    for scenario, v1_base, v2_base in SOURCES:
        for p in sorted(v1_base.iterdir()):
            if not p.is_dir():
                continue
            folder, ver = resolve_folder(p.name, v1_base, v2_base)
            lines = load_lines(folder)
            sessions.append((scenario, p.name, ver, lines,
                             guess_roles(lines)))

    counts = {key: {} for key, *_ in PHENOMENA}
    events_by_session = {name: [] for _, name, *_ in sessions}

    # 현상별 컬렉션 파일
    for key, desc, det, before, after in PHENOMENA:
        chunks = [f"# {key.split('_', 1)[1]} 컬렉션\n",
                  f"{desc}\n",
                  "표기: `(n.n)` 침묵(초) · `(.)` 미세 휴지 · "
                  "`∆…∆` 빠른 발화 · `∇…∇` 느린 발화 · ▶ 초점 턴\n"]
        total = 0
        for scenario, name, ver, lines, roles in sessions:
            hits = det(lines, roles)
            counts[key][name] = len(hits)
            if not hits:
                continue
            total += len(hits)
            chunks.append(f"\n## [{scenario}] {name}  ({ver}, "
                          f"튜터={next((s for s, r in roles.items() if r == '튜터'), '?')})\n")
            for focal, note in hits:
                t = next(l.start for l in lines if l.num == focal)
                chunks.append(f"\n**{fmt_time(t)}**"
                              + (f" — {note}" if note else "") + "\n")
                chunks.append("```")
                chunks.append(render_excerpt(lines, roles, focal,
                                             before, after))
                chunks.append("```")
                events_by_session[name].append(
                    (t, key.split("_", 1)[1], focal, note))
        chunks.insert(3, f"\n총 {total}건\n")
        (OUT / f"{key}.md").write_text("\n".join(chunks), encoding="utf-8")

    # 세션 궤적 지도
    for scenario, name, ver, lines, roles in sessions:
        ev = sorted(events_by_session[name])
        turns = [l for l in lines if not l.is_silence]
        dur = lines[-1].end if lines else 0
        role_str = ", ".join(f"{s}={r}" for s, r in sorted(roles.items()))
        rows = [f"# {name} 세션 궤적  [{scenario}/{ver}]\n",
                f"길이 {fmt_time(dur)} · 발화 턴 {len(turns)}개 · {role_str}\n",
                f"사건 {len(ev)}건 (시간순):\n"]
        for t, kind, focal, note in ev:
            line = next(l for l in lines if l.num == focal)
            snippet = (line.clean or line.text.strip())[:40]
            rows.append(f"- **{fmt_time(t)}** {kind} (행 {focal})"
                        + (f" — {note}" if note else "")
                        + (f" · “{snippet}…”" if snippet else ""))
        (OUT / "maps" / f"{name}.md").write_text("\n".join(rows),
                                                 encoding="utf-8")

    # INDEX
    keys = [k for k, *_ in PHENOMENA]
    rows = ["# CA 컬렉션 색인\n",
            "기계는 후보를 수집했을 뿐이다 — 각 사례가 실제 해당 현상인지, "
            "어떤 순차적 패턴을 이루는지는 발췌를 읽고 판정할 것.\n",
            "**한계**: Whisper 모노 전사는 겹침말을 기록하지 못한다 "
            "(코퍼스 전체 overlap 0건). '끼어들기후보'는 빠른 발화권 "
            "이동으로 잡은 대리 지표이므로, 실제 겹침 여부는 발췌의 "
            "타임스탬프로 원본 음성을 들어 확인해야 한다.\n",
            "| 세션 | 판 | " + " | ".join(k.split("_", 1)[1] for k in keys)
            + " |",
            "|---|---|" + "---|" * len(keys)]
    for scenario, name, ver, lines, roles in sessions:
        rows.append(f"| {name} | {ver} | "
                    + " | ".join(str(counts[k].get(name, 0)) for k in keys)
                    + " |")
    tot = ["| **합계** | | "
           + " | ".join(str(sum(counts[k].values())) for k in keys) + " |"]
    (OUT / "INDEX.md").write_text("\n".join(rows + tot), encoding="utf-8")

    print(f"세션 {len(sessions)}개 처리")
    for k in keys:
        print(f"  {k}: {sum(counts[k].values())}건")
    print(f"출력: {OUT}")


if __name__ == "__main__":
    main()
