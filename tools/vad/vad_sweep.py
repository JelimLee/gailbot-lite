# -*- coding: utf-8 -*-
"""VAD 임계값 스윕: 설정별 단어 목록을 저장해 '무엇을 잃는지' 비교한다.

단어 총수는 손실을 감춘다(어디선 늘고 어디선 사라짐). 그래서 각 설정의
결과를 모두 저장한 뒤, 전체 합집합 대비 커버리지로 평가한다.
"""
import csv
import time
from pathlib import Path
from faster_whisper import WhisperModel

AUDIO = f"{Path.home()}/Downloads/01_게일봇연구/gailbot_wav/CB001_260409_이혁건.wav"
OUT = Path(f"{Path.home()}/Downloads/01_게일봇연구/gailbot_analysis/vad_sweep")
THRESHOLDS = [0.40, 0.45]        # 0.35/0.50은 기존 v2/v1 결과 재사용

OUT.mkdir(parents=True, exist_ok=True)
model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")

for th in THRESHOLDS:
    t0 = time.time()
    segments, _ = model.transcribe(
        AUDIO, language="ko", word_timestamps=True,
        vad_filter=True, vad_parameters={"threshold": th})
    rows = [(f"{w.start:.2f}", w.word.strip())
            for seg in segments for w in (seg.words or []) if w.word.strip()]
    path = OUT / f"th{int(th*100)}.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["START TIME", "TEXT"])
        wr.writerows((s, t) for s, t in rows)
    print(f"threshold {th}: {len(rows)}단어, {time.time()-t0:.0f}초 → {path}",
          flush=True)

print("완료")
