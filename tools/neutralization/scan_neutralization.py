# -*- coding: utf-8 -*-
"""2024-25 ChatGPT 실험 엑셀 전사 100건에서 중립화(neutralization) 후보 스캔.

목적: 정밀 전사(게일봇) 없이도, 나머지 90건에 분석할 사례가 있는지 확인하고
세션별 밀도 순위를 매겨 '어느 세션부터 정밀 전사할지' 우선순위를 만든다.

탐지 항목 (프로포절 + 코드북 기반, 영/한 병용)
  요청  : U의 입장 요청 (what do you think / which side / 어떻게 생각해)
  중립화: G의 감사서두·양비론·반문 되돌리기 (great point / pros and cons /
          it depends / 둘 다 / 장단점 / what's your take)
  저항  : U의 양자택일 강요·반문 재이양·AI 정체성 소환·재추궁

출력: ~/Downloads/01_게일봇연구/gailbot_analysis/neutralization_scan.md
"""
import glob
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

SRC = f"{Path.home()}/Downloads/1 전사 엑셀 파일/*.xlsx"
OUT = Path(f"{Path.home()}/Downloads/01_게일봇연구/gailbot_analysis/neutralization_scan.md")
NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"

PAT = {
    "요청": re.compile(
        r"what do you think|what('s| is) your (opinion|take|view|stance)|"
        r"do you (agree|think|prefer)|which (side|one) (do|would) you|"
        r"your opinion|어떻게 생각|네 생각|너의? 생각|너는 어때|어느 쪽",
        re.I),
    "중립화": re.compile(
        r"great (point|question)|good (point|question)|interesting "
        r"(point|question)|both sides?|pros and cons|it depends|"
        r"depends on|on the other hand|balanced|neutral|"
        r"좋은 (질문|지적|포인트)|둘 다|양쪽|장단점|일장일단|"
        r"~?에 달려|나름이|중립", re.I),
    "반문": re.compile(
        r"what (do you think|about you|'s your take)|how about you|"
        r"your thoughts\??|당신의? 생각은|너는 어떻게 생각|어떻게 생각하세요",
        re.I),
    "저항_양자택일": re.compile(
        r"(which|choose|pick) one|a or b|if you (had|have) to (choose|pick)|"
        r"one of them|둘 중|하나만? (골라|선택|말해)|택일", re.I),
    "저항_재이양": re.compile(
        r"you (first|tell me first)|i asked you|i wan(t|na) to hear "
        r"(first )?from you|answer (my|the) question|네가 먼저|너부터|"
        r"내가 물어봤|먼저 말해|대답을 해", re.I),
    "저항_AI소환": re.compile(
        r"as an ai|you('re| are) an? ai|you know (everything|better|more)|"
        r"ai(니까|잖아|인데)|인공지능(이니까|이잖아)|다 알잖아", re.I),
}


def load_rows(path):
    """(화자, 발화) 목록. 열 위치는 헤더('화자','발화내용')로 찾는다."""
    z = zipfile.ZipFile(path)
    shared = []
    try:
        root = ET.fromstring(z.read("xl/sharedStrings.xml"))
        shared = ["".join(t.text or "" for t in si.iter(NS + "t"))
                  for si in root]
    except KeyError:
        pass
    sheet = ET.fromstring(z.read("xl/worksheets/sheet1.xml"))

    def cells_of(row):
        out = {}
        for c in row.iter(NS + "c"):
            ref = c.get("r", "")
            col = "".join(ch for ch in ref if ch.isalpha())
            v = c.find(NS + "v")
            val = v.text if v is not None else ""
            if c.get("t") == "s" and val:
                val = shared[int(val)]
            out[col] = val
        return out

    rows = [cells_of(r) for r in sheet.iter(NS + "row")]
    spk_col = txt_col = None
    for r in rows[:5]:
        for col, val in r.items():
            if val == "화자":
                spk_col = col
            if val == "발화내용":
                txt_col = col
    if not spk_col or not txt_col:
        return []
    out = []
    for r in rows:
        sp, tx = r.get(spk_col, ""), r.get(txt_col, "")
        if sp in ("U", "G", "DG") and tx:
            out.append((sp, tx))
    return out


def scan(rows):
    hits = {k: [] for k in PAT}
    for i, (sp, tx) in enumerate(rows):
        for key, rx in PAT.items():
            who = "U" if (key == "요청" or key.startswith("저항")) else "G"
            if sp == who or (who == "G" and sp == "DG"):
                m = rx.search(tx)
                if m:
                    hits[key].append((i, m.group(0)))
    return hits


def main():
    files = sorted(glob.glob(SRC))
    results = []
    for p in files:
        rows = load_rows(p)
        if not rows:
            results.append((Path(p).stem, 0, {}, [], 0))
            continue
        hits = scan(rows)
        resist = sum(len(v) for k, v in hits.items() if k.startswith("저항"))
        # 밀도 점수: 저항이 가장 희귀하고 분석 가치가 높으므로 가중
        score = (len(hits["요청"]) + len(hits["중립화"]) + len(hits["반문"])
                 + 3 * resist)
        results.append((Path(p).stem, score, hits, rows, resist))

    results.sort(key=lambda r: -r[1])

    md = ["# 중립화 스캔 — 엑셀 러프 전사 100건\n",
          "러프 전사 텍스트만으로 잡은 후보이므로 건수는 하한선이다 "
          "(휴지·비언어 신호는 게일봇 정밀 전사에서만 보인다).\n",
          "| 순위 | 세션 | 점수 | 요청 | 중립화 | 반문 | 저항(3종) |",
          "|---|---|---|---|---|---|---|"]
    n_any = 0
    for rank, (name, score, hits, rows, resist) in enumerate(results, 1):
        if score > 0:
            n_any += 1
        if rank <= 40:
            md.append(f"| {rank} | {name} | {score} | "
                      f"{len(hits.get('요청', []))} | "
                      f"{len(hits.get('중립화', []))} | "
                      f"{len(hits.get('반문', []))} | {resist} |")

    md.append(f"\n**후보가 1건이라도 있는 세션: {n_any}/{len(results)}**\n")
    md.append("\n## 상위 세션 발췌 (저항 사례 중심)\n")
    shown = 0
    for name, score, hits, rows, resist in results:
        if shown >= 8 or resist == 0:
            continue
        shown += 1
        md.append(f"\n### {name} (저항 {resist}건)\n")
        for key in ("저항_양자택일", "저항_재이양", "저항_AI소환"):
            for i, matched in hits.get(key, [])[:2]:
                md.append(f"\n**{key}** — “{matched}”\n")
                md.append("```")
                for j in range(max(0, i - 2), min(len(rows), i + 3)):
                    mark = "▶" if j == i else " "
                    sp, tx = rows[j]
                    md.append(f" {mark} {sp}: {tx[:100]}")
                md.append("```")

    OUT.write_text("\n".join(md), encoding="utf-8")
    print(f"스캔 완료: {len(results)}개 세션")
    print(f"중립화 관련 후보가 있는 세션: {n_any}개")
    top = results[0]
    print(f"최고 밀도: {top[0]} (점수 {top[1]})")
    print(f"리포트: {OUT}")


if __name__ == "__main__":
    main()
