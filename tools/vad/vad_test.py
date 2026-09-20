# -*- coding: utf-8 -*-
"""CB001 한 파일로 VAD 설정 후보를 비교해 누락이 가장 적은 조합을 찾는다.

전체 배치(3시간)를 돌리기 전에 설정을 검증하는 용도.
화자 분리는 생략(설정과 무관하고 느림) — 단어 인식률만 본다.
"""
from pathlib import Path
import time
from faster_whisper import WhisperModel

AUDIO = f"{Path.home()}/Downloads/01_게일봇연구/gailbot_wav/CB001_260409_이혁건.wav"

# v1에서 확인된 실제 발화. 설정별로 살아남는지 본다.
PROBES = {
    "진행자 안내(1.6s~)": "편하신",
    "사용자 턴(330s~)": "창출하는",
    "챗봇 첫 발화(63s~)": "받았어요",
}

CONFIGS = {
    "A_v1기본": dict(vad_filter=True),
    "B_현재v2": dict(vad_filter=True,
                    vad_parameters={"threshold": 0.25, "speech_pad_ms": 700,
                                    "min_silence_duration_ms": 2000},
                    condition_on_previous_text=False),
    "C_VAD끔+환청가드": dict(vad_filter=False,
                            condition_on_previous_text=False,
                            hallucination_silence_threshold=2.0),
    "D_임계값만완화": dict(vad_filter=True,
                          vad_parameters={"threshold": 0.35}),
}

model = WhisperModel("large-v3-turbo", device="cpu", compute_type="int8")
results = {}

for label, kw in CONFIGS.items():
    t0 = time.time()
    segments, _ = model.transcribe(AUDIO, language="ko",
                                   word_timestamps=True, **kw)
    words = [(float(w.start), w.word.strip())
             for seg in segments for w in (seg.words or [])]
    text = " ".join(w for _, w in words)
    found = {k: ("O" if p in text else "X") for k, p in PROBES.items()}
    results[label] = (len(words), found, time.time() - t0)
    print(f"[{label}] {len(words)}단어, {time.time()-t0:.0f}초  "
          + " ".join(f"{k}:{v}" for k, v in found.items()), flush=True)

print("\n" + "=" * 70)
print(f"{'설정':<18}{'단어수':>8}  " + "  ".join(PROBES))
for label, (n, found, dt) in results.items():
    print(f"{label:<18}{n:>8}  " + "  ".join(
        f"{found[k]:^{len(k)}}" for k in PROBES))
