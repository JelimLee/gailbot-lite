# -*- coding: utf-8 -*-
"""설정별 커버리지 비교: 어느 임계값이 가장 적게 잃는가.

10초 버킷 단위로 각 설정의 단어 수를 비교한다. 어떤 설정에는 발화가 있고
다른 설정에는 없는 구간 = 그 설정이 잃은 구간.
"""
import csv
from pathlib import Path

SWEEP = Path(f"{Path.home()}/Downloads/01_게일봇연구/gailbot_analysis/vad_sweep")
NAME = "CB001_260409_이혁건"
SOURCES = {
    "th35": Path(f"{Path.home()}/Downloads/01_게일봇연구/gailbot_results_v2") / NAME
            / "conversation_words.csv",
    "th40": SWEEP / "th40.csv",
    "th45": SWEEP / "th45.csv",
    "th50": Path(f"{Path.home()}/Downloads/01_게일봇연구/gailbot_results") / NAME
            / "conversation_words.csv",
}
BUCKET = 10.0


def load(path):
    b = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            t = float(row["START TIME"])
            b.setdefault(int(t // BUCKET), []).append(row["TEXT"])
    return b


data = {k: load(p) for k, p in SOURCES.items() if p.exists()}
keys = list(data)
all_buckets = sorted(set().union(*(set(d) for d in data.values())))

print(f"파일: {NAME}   버킷 {BUCKET:.0f}초   설정 {keys}\n")
for k in keys:
    print(f"  {k}: 총 {sum(len(v) for v in data[k].values())}단어, "
          f"발화 버킷 {len(data[k])}개")

# 각 설정이 '혼자만 놓친' 구간
print("\n=== 설정별 누락 구간 (다른 설정엔 있는데 이 설정엔 없음) ===")
loss = {k: [] for k in keys}
for b in all_buckets:
    present = [k for k in keys if data[k].get(b)]
    missing = [k for k in keys if not data[k].get(b)]
    if not missing or not present:
        continue
    # 다른 설정에서 실제로 유의미한 발화가 있었던 구간만
    ref = max(present, key=lambda k: len(data[k][b]))
    if len(data[ref][b]) < 3:
        continue
    for k in missing:
        loss[k].append((b, " ".join(data[ref][b])[:70]))

for k in keys:
    print(f"\n[{k}] 누락 구간 {len(loss[k])}개")
    for b, txt in loss[k][:6]:
        print(f"   {b*BUCKET:>6.0f}s  {txt}")

print("\n=== 종합 ===")
print(f"{'설정':<8}{'단어수':>8}{'누락구간':>10}")
for k in sorted(keys, key=lambda k: (len(loss[k]),
                                     -sum(len(v) for v in data[k].values()))):
    print(f"{k:<8}{sum(len(v) for v in data[k].values()):>8}{len(loss[k]):>10}")
