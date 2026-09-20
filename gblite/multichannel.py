# -*- coding: utf-8 -*-
"""다중 채널 녹음에서 진짜 겹침(overlap) 추출.

배경: 지금까지의 녹음은 좌우 채널이 동일한 dual-mono여서(상관계수
0.9998~1.0000 실측) 겹침말이 원리적으로 복원되지 않았다. 다음 실험부터
화자별로 마이크를 분리해 녹음하면, 각 채널을 따로 전사한 뒤 시간축에서
겹치는 구간을 찾아 Jefferson 표기의 대괄호를 붙일 수 있다.

두 가지 입력 형태를 받는다.
  1) 화자별 파일이 따로 있는 경우  — 파일 하나 = 화자 하나
  2) 한 파일에 채널이 분리된 경우  — 채널 하나 = 화자 하나 (여기서 분리)

주의: 마이크가 분리돼도 상대 목소리가 새어 들어온다(crosstalk). 그대로
전사하면 두 채널에 같은 말이 잡히므로, 채널 간 에너지 비교로 걸러낸다.
"""
import os
import subprocess
import tempfile
from typing import Dict, List, Optional, Tuple

from .models import Word, Marker


def split_channels(audio_path: str, out_dir: Optional[str] = None
                   ) -> List[str]:
    """다채널 파일을 채널별 wav로 분리한다. 반환: 채널 파일 경로 목록."""
    n = probe_channels(audio_path)
    if n < 2:
        return [audio_path]
    out_dir = out_dir or tempfile.mkdtemp(prefix="gblite_ch_")
    stem = os.path.splitext(os.path.basename(audio_path))[0]
    outs = [os.path.join(out_dir, f"{stem}_ch{i+1}.wav") for i in range(n)]
    # 채널 분리 + 16kHz 모노로 통일
    filt = f"channelsplit=channel_layout={'stereo' if n == 2 else f'{n}c'}"
    cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-i", audio_path,
           "-filter_complex", filt]
    for i, o in enumerate(outs):
        cmd += ["-map", f"[{_layout_label(n, i)}]", "-ar", "16000", "-ac", "1",
                "-y", o]
    subprocess.run(cmd, capture_output=True)
    return [o for o in outs if os.path.exists(o)]


def _layout_label(n: int, i: int) -> str:
    if n == 2:
        return ["FL", "FR"][i]
    return f"c{i}"


def probe_channels(audio_path: str) -> int:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=channels", "-of", "csv=p=0", audio_path],
        capture_output=True, text=True)
    try:
        return int((r.stdout or "0").strip().split("\n")[0])
    except (ValueError, IndexError):
        return 0


def channels_are_distinct(paths: List[str], thresh: float = 0.98) -> bool:
    """채널이 실제로 다른 소리인지 확인. dual-mono면 False.

    분리 녹음이라고 믿고 돌렸는데 실은 같은 신호인 경우를 잡는다."""
    if len(paths) < 2:
        return False
    try:
        import numpy as np
        import wave

        def read(p):
            with wave.open(p) as w:
                d = np.frombuffer(w.readframes(w.getnframes()),
                                  dtype=np.int16).astype(float)
            return d
        a, b = read(paths[0]), read(paths[1])
        n = min(len(a), len(b))
        if n == 0:
            return False
        c = np.corrcoef(a[:n], b[:n])[0, 1]
        return abs(c) < thresh
    except Exception:
        return True          # 판정 불가면 진행시킨다


def suppress_crosstalk(per_channel: Dict[str, List[Word]],
                       min_ratio: float = 1.5) -> Dict[str, List[Word]]:
    """다른 채널로 새어 들어온 말을 제거한다.

    같은 시간대에 여러 채널이 거의 같은 단어를 잡았다면, 그 구간의
    에너지가 가장 큰 채널만 남긴다. 에너지 정보가 없으므로 여기서는
    **단어 수가 더 많은 쪽(= 더 온전히 인식된 쪽)**을 원 화자로 본다."""
    labels = list(per_channel)
    if len(labels) < 2:
        return per_channel
    out = {k: [] for k in labels}
    for lab in labels:
        for w in per_channel[lab]:
            dup = False
            for other in labels:
                if other == lab:
                    continue
                for o in per_channel[other]:
                    if o.text != w.text:
                        continue
                    # 시간이 거의 같으면 중복으로 본다
                    if abs(o.start - w.start) < 0.25 and \
                            abs(o.end - w.end) < 0.25:
                        if len(per_channel[other]) > len(per_channel[lab]):
                            dup = True
                        break
                if dup:
                    break
            if not dup:
                out[lab].append(w)
    return out


def merge_words(per_channel: Dict[str, List[Word]]) -> List[Word]:
    """채널별 단어를 하나의 시간순 목록으로 합친다(화자 라벨 유지)."""
    merged: List[Word] = []
    for spk, words in per_channel.items():
        for w in words:
            merged.append(Word(start=w.start, end=w.end, text=w.text,
                               speaker=spk, provenance=w.provenance,
                               tail_candidate=w.tail_candidate))
    merged.sort(key=lambda w: (w.start, w.end))
    return merged


def detect_true_overlaps(per_channel: Dict[str, List[Word]],
                         min_dur: float = 0.05) -> List[Marker]:
    """채널이 분리됐을 때만 가능한 **실제 동시 발화** 검출.

    기존 detect_overlaps는 발화(utterance) 단위로 시간이 겹치는지를 보는데,
    모노 전사에서는 단어가 순차적으로만 나오므로 사실상 잡히지 않았다.
    채널이 분리되면 두 화자의 단어가 같은 시각에 존재할 수 있으므로,
    여기서 단어 단위로 겹침 구간을 직접 찾는다."""
    labels = sorted(per_channel)
    markers: List[Marker] = []
    uid = 0
    for i, a_lab in enumerate(labels):
        for b_lab in labels[i + 1:]:
            for wa in per_channel[a_lab]:
                for wb in per_channel[b_lab]:
                    if wb.start >= wa.end:
                        break
                    if wb.end <= wa.start:
                        continue
                    s = max(wa.start, wb.start)
                    e = min(wa.end, wb.end)
                    if e - s < min_dur:
                        continue
                    markers.append(Marker(s, s, "overlap_first_start",
                                          str(uid), a_lab))
                    markers.append(Marker(e, e, "overlap_first_end",
                                          str(uid), a_lab))
                    markers.append(Marker(s, s, "overlap_second_start",
                                          str(uid), b_lab))
                    markers.append(Marker(e, e, "overlap_second_end",
                                          str(uid), b_lab))
                    uid += 1
    markers.sort(key=lambda m: (m.start, m.end))
    return markers
