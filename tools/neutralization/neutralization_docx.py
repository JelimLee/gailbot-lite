# -*- coding: utf-8 -*-
"""중립화 시퀀스 전량을 연도별 docx로 출력.

CA 기준에 맞춰 **위치**로 자른다.
  1. 개시   : 중립화를 촉발한 앞 턴 (사용자의 입장 요청·저항)
  2. 중립화 : GPT가 입장을 유보하는 턴
  3. 응답   : 바로 다음 사용자 턴 — 여기가 저항인지 정렬인지 드러나는 자리
  4. 처리   : 그에 대한 GPT의 다음 턴

저항은 표현이 아니라 **자리**로 정의한다. "recommend me…"가 대화 첫머리에
나오면 단순 요청이고, 중립화 다음 자리에 나와야 저항이다.

출력
  ~/Downloads/01_게일봇연구/고베 학회 준비/중립화_시퀀스_2025.docx
  ~/Downloads/01_게일봇연구/고베 학회 준비/중립화_시퀀스_2026.docx
"""
import csv
import html
import glob
import re
import subprocess
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

DL = Path.home() / "Downloads"
OUT = DL / "고베 학회 준비"
CA2025 = DL / "0 ChatGPT4o 전사전환 완료"
XLSX = DL / "1 전사 엑셀 파일"
GB2026 = DL / "gailbot_results_gpt2026"
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

APPRECIATE = re.compile(
    r"that'?s (a )?(great|good|interesting|really good)|great (point|question)|"
    r"good (point|question)|interesting (point|question)|"
    r"좋은 (질문|지적|포인트)|훌륭한 질문", re.I)
BOTHSIDES = re.compile(
    r"both (sides?|are|can|have)|pros and cons|it depends|depends on|"
    r"on the other hand|there are (valid|good) (points|arguments)|balanced|"
    r"각각의? 장단점|장단점이|둘 다 (좋|맞|가능|장점|중요)|양쪽 (다|모두)|"
    r"경우에 따라|상황에 따라|나름의? 장점|일장일단|어느 한쪽|정답은 없|"
    r"사람마다 다르|반대로 .{0,40}(수도|것 같|겠지|맞겠)|"
    r"반면 .{0,40}(수도|것 같|겠지|유리)", re.I)
RETURNQ = re.compile(
    r"what do you think|what'?s your (take|view|opinion)|how about you|"
    r"your thoughts|what would you say|"
    r"어떻게 생각(해|하세요|하시나요)|네 생각은|당신은 어때", re.I)
DEFER = re.compile(
    r"ultimately (it'?s )?up to you|it'?s (your|a personal) (choice|decision)|"
    r"결국 (네|너의|당신의)|네가 (원하|중시|판단|결정|선택)|"
    r"(선택|결정)하(면 돼|면 좋을|시면 돼)|본인이 (판단|결정)", re.I)

RESIST = {
    "① 양자택일 강요": re.compile(
        r"if you (had|have) to (choose|pick)|(choose|pick) (one|a side)|"
        r"which (one|side) (do|would)|just one|only one|"
        r"둘 중 (하나|어느)|하나만? (골라|고르|선택|정해)|골라\s*줘|"
        r"어느 쪽(이야|인지|을)|뭐가 (더|좀 더)", re.I),
    "② 반문 되돌리기": re.compile(
        r"i (wan(t|na)|would like) to hear (first )?from you|you (go )?first|"
        r"i asked you|answer (my|the) question|i'?m asking you|"
        r"네가 먼저|너부터|내가 물어봤|먼저 (말해|대답)|너라면|너였으면|"
        r"너 같으면", re.I),
    "③ AI 정체성 소환": re.compile(
        r"as an ai|you'?re an ai|you are an ai|you know (everything|more|better)|"
        r"ai(니까|잖아|인데|라면)|인공지능(이니까|이잖아|인데)|넌 다 알잖아", re.I),
    "④ 유보의 부적절성 지적": re.compile(
        r"that'?s not (what|an answer|helpful|the point)|doesn'?t (help|answer)|"
        r"not helpful|too (general|abstract|vague|neutral)|you didn'?t answer|"
        r"i want you to (say|take|choose)|no+,? i want|"
        r"너무 (추상적|원론적|일반적)|도움이 (안|잘 안) (돼|되)|뻔한|"
        r"그런 (거|얘기) 말고", re.I),
    "⑤ 권고 요구": re.compile(
        r"what should i (do|choose|pick)|tell me what (to do|you'?d)|"
        r"give me (a |your )?(recommendation|advice|answer)|recommend me|"
        r"just tell me|추천해\s*(줘|줄래|주라)|제시해\s*(줘|줄래)|"
        r"정해\s*(줘|줄래)|어떻게 해야 (돼|해)", re.I),
}
OPINION = re.compile(
    r"^\s*(i think|i believe|in my opinion|personally|i'?d say|i agree|"
    r"i feel)|나는 |내 생각에|제 생각(엔|에는)|나도 ", re.I)

SPEAK_RE = re.compile(r"^\s*([UGARB])\s*(?:[:：]|\t)\s*(.*)$")


class T:
    def __init__(self, n, sp, text):
        self.n, self.sp, self.text = n, sp, text


# ── 읽기 ─────────────────────────────────────────────────────────────────
def from_doc(path):
    if path.suffix.lower() == ".doc":
        r = subprocess.run(["textutil", "-convert", "txt", "-stdout",
                            str(path)], capture_output=True)
        paras = r.stdout.decode("utf-8", errors="ignore").split("\n")
    else:
        xml = zipfile.ZipFile(path).read("word/document.xml").decode("utf-8")
        paras = [html.unescape(re.sub(r"<[^>]+>", "", p))
                 for p in re.findall(r"<w:p[ >].*?</w:p>", xml, flags=re.S)]
    turns, n = [], 0
    for p in paras:
        if not p.strip():
            continue
        m = SPEAK_RE.match(p)
        if m:
            n += 1
            turns.append(T(n, m.group(1), m.group(2).strip()))
        elif turns:
            turns[-1].text += " " + p.strip()
    return turns


def from_xlsx(path):
    z = zipfile.ZipFile(path)
    sh = []
    try:
        sh = ["".join(t.text or "" for t in si.iter(NS + "t"))
              for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
    except KeyError:
        pass
    st = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))
    rows = []
    for row in st.iter(NS + "row"):
        c = {}
        for cell in row.iter(NS + "c"):
            col = "".join(ch for ch in cell.get("r", "") if ch.isalpha())
            v = cell.find(NS + "v")
            val = v.text if v is not None else ""
            if cell.get("t") == "s" and val:
                val = sh[int(val)]
            c[col] = val
        rows.append(c)
    sp = tx = None
    for r in rows[:5]:
        for col, val in r.items():
            if val == "화자":
                sp = col
            if val == "발화내용":
                tx = col
    turns, n = [], 0
    if not (sp and tx):
        return turns
    for r in rows:
        s, t = r.get(sp, ""), r.get(tx, "")
        if s in ("U", "G", "DG") and t:
            n += 1
            turns.append(T(n, "G" if s in ("G", "DG") else "U", t))
    return turns


def from_gailbot(folder):
    rows = list(csv.DictReader(
        open(folder / "conversation_turns.csv", encoding="utf-8")))
    sc = {}
    for r in rows:
        s = (r.get("SPEAKER LABEL") or "").strip()
        if not s:
            continue
        t = r["TEXT"] or ""
        v = sc.setdefault(s, [0, 0])
        v[0] += len(BOTHSIDES.findall(t)) + len(DEFER.findall(t)) \
            + len(APPRECIATE.findall(t))
        v[1] += len(t.split())
    if not sc:
        return []
    g = max(sc, key=lambda s: (sc[s][0], sc[s][1]))
    turns, n = [], 0
    for r in rows:
        s = (r.get("SPEAKER LABEL") or "").strip()
        t = (r["TEXT"] or "").strip()
        if not (s and t):
            continue
        n += 1
        turns.append(T(n, "G" if s == g else "U", t))
    return turns


# ── 시퀀스 ───────────────────────────────────────────────────────────────
def neutral_marks(t):
    m = []
    if APPRECIATE.search(t):
        m.append("인정 서두")
    if BOTHSIDES.search(t):
        m.append("양비론")
    if RETURNQ.search(t):
        m.append("질문 되돌리기")
    if DEFER.search(t):
        m.append("판단 되넘기기")
    return m


def classify_response(text):
    """중립화 **다음 자리**의 사용자 턴이 무엇을 하는가."""
    for label, rx in RESIST.items():
        m = rx.search(text)
        if m:
            return f"저항 {label}", m.group(0)
    if OPINION.search(text):
        return "정렬 — 사용자가 자기 의견을 이어감", ""
    return "기타", ""


def sequences(turns):
    out = []
    for i, t in enumerate(turns):
        if t.sp != "G":
            continue
        marks = neutral_marks(t.text)
        if len(marks) < 2:
            continue
        nxt = next((j for j in range(i + 1, min(i + 3, len(turns)))
                    if turns[j].sp == "U"), None)
        resp, ev = classify_response(turns[nxt].text) if nxt else ("(없음)", "")
        lo = max(0, i - 1)
        hi = (nxt + 2) if nxt else i + 1
        out.append({"marks": marks, "resp": resp, "ev": ev,
                    "turns": turns[lo:min(hi, len(turns))], "focal": t.n})
    return out


# ── docx ─────────────────────────────────────────────────────────────────
def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def para(text, bold=False, size=18):
    rpr = (f'<w:rPr><w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" '
           f'w:eastAsia="맑은 고딕"/>{"<w:b/>" if bold else ""}'
           f'<w:sz w:val="{size}"/></w:rPr>')
    return (f'<w:p><w:pPr>{rpr}</w:pPr><w:r>{rpr}'
            f'<w:t xml:space="preserve">{esc(text)}</w:t></w:r></w:p>')


CT = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<Types xmlns='
      '"http://schemas.openxmlformats.org/package/2006/content-types">'
      '<Default Extension="rels" ContentType="application/vnd.openxmlformats-'
      'package.relationships+xml"/><Default Extension="xml" ContentType='
      '"application/xml"/><Override PartName="/word/document.xml" ContentType='
      '"application/vnd.openxmlformats-officedocument.wordprocessingml.'
      'document.main+xml"/></Types>')
RELS = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/'
        'relationships"><Relationship Id="rId1" Type="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships/officeDocument"'
        ' Target="word/document.xml"/></Relationships>')
DOC = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n<w:document '
       'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
       '<w:body>{b}<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar '
       'w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
       '</w:sectPr></w:body></w:document>')


def write_docx(path, paras):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", CT)
        z.writestr("_rels/.rels", RELS)
        z.writestr("word/document.xml", DOC.format(b="".join(paras)))


def wrap(t, w=62):
    ws, cur, out = t.split(), "", []
    for x in ws:
        if cur and len(cur) + len(x) + 1 > w:
            out.append(cur)
            cur = x
        else:
            cur = f"{cur} {x}".strip()
    out.append(cur)
    return out


def build(year, sources, title_note):
    ps = [para(f"중립화 시퀀스 — {year}", bold=True, size=28), para("")]
    for line in title_note:
        ps.append(para(line))
    ps.append(para(""))
    total = 0
    for label, name, turns in sources:
        seqs = sequences(turns)
        if not seqs:
            continue
        ps.append(para(""))
        ps.append(para(f"── {name}  [{label}]  총 {len(turns)}턴 · "
                       f"중립화 {len(seqs)}건", bold=True, size=22))
        for s in seqs:
            total += 1
            ps.append(para(""))
            ps.append(para(f"[{total}] {s['focal']}행 — "
                           + " + ".join(s["marks"]), bold=True))
            ps.append(para(f"     다음 자리: {s['resp']}"
                           + (f" “{s['ev']}”" if s["ev"] else "")))
            for t in s["turns"]:
                mark = "→" if t.n == s["focal"] else " "
                lines = wrap(t.text)
                ps.append(para(f"{t.n:>4} {mark} {t.sp}: {lines[0]}"))
                for l in lines[1:]:
                    ps.append(para(f"          {l}"))
    ps.insert(2, para(f"총 {total}건", bold=True))
    return ps, total


def main():
    OUT.mkdir(exist_ok=True)

    src25 = []
    seen = set()
    for p in sorted(CA2025.glob("*")):
        if p.suffix.lower() in (".doc", ".docx"):
            key = re.sub(r"[^가-힣A-Za-z]", "", p.stem)[:8]
            seen.add(key)
            src25.append(("수작업 CA 전사본", p.stem, from_doc(p)))
    for p in sorted(glob.glob(str(XLSX / "*.xlsx"))):
        stem = Path(p).stem
        if re.sub(r"[^가-힣A-Za-z]", "", stem)[:8] in seen:
            continue
        src25.append(("엑셀 러프 전사", stem, from_xlsx(Path(p))))

    src26 = [("게일봇 자동 전사", f.name, from_gailbot(f))
             for f in sorted(GB2026.iterdir())
             if f.is_dir() and (f / "conversation_turns.csv").exists()]

    note25 = [
        "자료: 연구팀 수작업 CA 전사본(휴지·늘림·겹침 표기 있음) + 엑셀 러프",
        "전사(표기 없음). 세션마다 어느 쪽인지 표시했다.",
        "",
        "시퀀스는 위치로 잘랐다 — 중립화 앞 1턴(무엇이 촉발했나), 중립화 턴(→),",
        "바로 다음 사용자 턴(저항인가 정렬인가), 그에 대한 GPT의 처리.",
        "저항은 표현이 아니라 자리로 판정한다.",
    ]
    note26 = [
        "자료: 게일봇 자동 전사. 휴지 (n.n)·미세휴지 (.)·래칭 = 표기 있음.",
        "겹침은 없다 — 녹음이 dual-mono라 겹침말이 복원되지 않는다.",
        "",
        "2026 재실험은 과제 유형이 토론이 아니라 진로 상담·모의면접 중심이라,",
        "중립화가 '입장 유보'보다 '권고 유보' 형태로 나타난다.",
    ]

    for year, src, note in (("2025", src25, note25), ("2026", src26, note26)):
        ps, n = build(year, src, note)
        path = OUT / f"중립화_시퀀스_{year}.docx"
        write_docx(path, ps)
        print(f"{year}: 중립화 시퀀스 {n}건 → {path.name}")


if __name__ == "__main__":
    main()
