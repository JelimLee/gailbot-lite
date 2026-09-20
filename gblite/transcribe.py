# -*- coding: utf-8 -*-
"""
Speech-to-text via faster-whisper (local, no API key, Apple Silicon OK).

Replaces GailBot's engine layer (Watson/Whisper/Google) with a single
modern local engine. Word-level timestamps are enabled so the CA
analysis has the same granularity as the original pipeline.
"""
import csv
import os
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

from .models import Word


# ── 세그먼트 신뢰도 (Radford et al. 2022, Whisper 원 논문 4.5절) ──────────
#
# 원 논문은 compression_ratio > 2.4 (같은 말이 반복돼 텍스트가 지나치게 잘
# 압축된다 = 환각의 흔적) 또는 avg_logprob < -1.0 (모델이 스스로 확신하지
# 못한다) 인 세그먼트를 "신뢰하지 않는다"고 적었다.
#
# 여기서는 **표시만 한다. 절대 지우지 않는다.** 이 코퍼스의 어떤 수치도
# 사람 CA 전사본과 대조되기 전이고, 자동 삭제는 되돌릴 수 없다.
RADFORD_MAX_COMPRESSION_RATIO = 2.4     # 문헌: Radford et al. 2022 §4.5
RADFORD_MIN_AVG_LOGPROB = -1.0          # 문헌: Radford et al. 2022 §4.5

SEGMENT_CSV_FIELDS = [
    "SOURCE", "PASS", "SEG ID", "START TIME", "END TIME", "TEXT",
    "AVG LOGPROB", "COMPRESSION RATIO", "NO SPEECH PROB",
    "RADFORD SUSPECT", "RADFORD REASON",
]

# 이 프로세스에서 디코딩한 세그먼트가 여기 쌓인다. transcribe_file() 은
# 낱말(List[Word])만 돌려주게 되어 있고 그 형식은 아래 파이프라인 전체가
# 의존하므로 반환형을 바꾸지 않는다. 대신 세그먼트 층을 옆으로 뺀다.
# 세션 경계에서 reset_segments() 로 반드시 비운다 — 안 비우면 앞 세션
# 세그먼트가 다음 세션 CSV 에 조용히 섞인다.
_SEGMENTS: List[Dict] = []


# ── 실행 조건 기록 (2026-08-21) ──────────────────────────────────────────
#
# `cpu_threads=_perf_cores()` 는 **기계마다 다른 값**이 된다(이 맥은 6).
# ctranslate2 커널은 스레드에 일감을 **정적으로 분할**하므로, 스레드 수가
# 달라지면 축약(reduction) 순서가 달라지고 부동소수점 비결합성 때문에
# 디코딩 결과가 갈릴 수 있다. (반대로 CPU 경합은 스케줄링만 흔들 뿐 분할을
# 바꾸지 않는다 — 6프로세스 부하 아래에서 본문 796낱말이 낱말 하나까지
# 같았다. `실측(GAC017 앞 10분, 2026-08-21)`)
#
# 그런데 이 값은 지금까지 **산출물 어디에도 남지 않았다.** 다른 맥에서 돌린
# 결과와 섞이면 나중에 알아낼 방법이 없다. 그래서 세션 산출물
# (`*_stats.json` 의 `run_config` 칸)에 적는다.
#
# **기록만 한다. 값을 고정하지 않는다** — 운영 배치가 도는 중일 수 있고,
# 배치 도중에 설정을 바꾸면 코퍼스가 두 설정으로 갈린다.
_RUN_CONFIG: Dict = {}
_RUN_PASSES: List[Dict] = []
_PERF_CORES_SOURCE = ""     # "sysctl" 이면 실측, "fallback" 이면 추정값


def reset_run_config() -> None:
    """세션 경계에서 비운다. 안 비우면 앞 세션 조건이 다음 세션에 새어 든다."""
    _RUN_CONFIG.clear()
    del _RUN_PASSES[:]


def run_config() -> Optional[Dict]:
    """이 프로세스가 **실제로** 돈 전사 조건. 전사를 안 거쳤으면 None.

    빈 dict 를 돌려주면 '조건이 비어 있음'과 '전사를 안 함'(annotate 경로)이
    구별되지 않는다. 그래서 None 이다 — 호출자가 아예 안 적는다.
    """
    if not _RUN_CONFIG:
        return None
    out = dict(_RUN_CONFIG)
    out["passes"] = [dict(x) for x in _RUN_PASSES]
    return out


def reset_segments() -> int:
    """누적된 세그먼트를 비우고, 버린 개수를 돌려준다(호출자가 찍을 수 있게)."""
    n = len(_SEGMENTS)
    _SEGMENTS.clear()
    return n


def collected_segments() -> List[Dict]:
    return list(_SEGMENTS)


def _radford_flag(avg_logprob, compression_ratio) -> Tuple[str, str]:
    """(suspect, reason). 필드가 없으면 판정하지 않고 'unknown' 으로 남긴다.

    없는 값을 0 이나 False 로 메우면 '멀쩡함'과 '못 쟀음'이 구별되지 않는다.
    """
    reasons = []
    known = 0
    if compression_ratio is not None:
        known += 1
        if compression_ratio > RADFORD_MAX_COMPRESSION_RATIO:
            reasons.append(f"compression_ratio>{RADFORD_MAX_COMPRESSION_RATIO}")
    if avg_logprob is not None:
        known += 1
        if avg_logprob < RADFORD_MIN_AVG_LOGPROB:
            reasons.append(f"avg_logprob<{RADFORD_MIN_AVG_LOGPROB}")
    if known == 0:
        return "unknown", "필드없음"
    return ("1" if reasons else "0"), "+".join(reasons)


def _record_segment(seg, source: str, pass_tag: str) -> None:
    """faster_whisper Segment 하나를 신뢰도 필드째 적어 둔다.

    걸러내지 않는다 — 빈 텍스트 세그먼트도 그대로 남긴다. 여기서 빼면
    낱말 CSV 와 세그먼트 CSV 의 분모가 달라져 교차표를 못 만든다.
    """
    alp = getattr(seg, "avg_logprob", None)
    cr = getattr(seg, "compression_ratio", None)
    nsp = getattr(seg, "no_speech_prob", None)
    suspect, reason = _radford_flag(alp, cr)
    _SEGMENTS.append({
        "SOURCE": source,
        "PASS": pass_tag,
        "SEG ID": getattr(seg, "id", len(_SEGMENTS)),
        "START TIME": "%.3f" % float(seg.start),
        "END TIME": "%.3f" % float(seg.end),
        "TEXT": (seg.text or "").strip(),
        # 자리수로 반올림하면 안 된다. no_speech_prob 는 3.8e-11 처럼
        # 아주 작은 값이 정상이라 round(,5) 를 먹이면 전부 0.0 이 되어
        # "무음 확률이 낮다"와 "값이 없다"가 구별되지 않는다
        # (2026-08-21 에 실제로 밟았다 — 115개 전부 0.0 으로 찍혔다).
        # 유효숫자 12자리로 적는다.
        "AVG LOGPROB": "" if alp is None else "%.12g" % float(alp),
        "COMPRESSION RATIO": "" if cr is None else "%.12g" % float(cr),
        "NO SPEECH PROB": "" if nsp is None else "%.12g" % float(nsp),
        "RADFORD SUSPECT": suspect,
        "RADFORD REASON": reason,
    })


def write_segments_csv(path: str, segments: Optional[List[Dict]] = None,
                       quiet: bool = False) -> Dict[str, int]:
    """세그먼트 신뢰도 CSV 를 쓰고, 총 개수와 의심 개수를 같은 자리에 찍는다."""
    segs = collected_segments() if segments is None else segments
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=SEGMENT_CSV_FIELDS)
        w.writeheader()
        for row in segs:
            w.writerow(row)
    n = len(segs)
    n_susp = sum(1 for r in segs if r["RADFORD SUSPECT"] == "1")
    n_unk = sum(1 for r in segs if r["RADFORD SUSPECT"] == "unknown")
    n_cr = sum(1 for r in segs if "compression_ratio" in r["RADFORD REASON"])
    n_lp = sum(1 for r in segs if "avg_logprob" in r["RADFORD REASON"])
    if not quiet:
        pct = f"{n_susp / n:.1%}" if n else "—"
        print(f"    세그먼트 {n}개 · Radford 의심 {n_susp}개({pct}, 분모 {n}) "
              f"[compression_ratio>{RADFORD_MAX_COMPRESSION_RATIO} {n_cr}개 · "
              f"avg_logprob<{RADFORD_MIN_AVG_LOGPROB} {n_lp}개 · "
              f"판정불가 {n_unk}개] — 플래그만, 제거하지 않음")
    return {"n": n, "suspect": n_susp, "unknown": n_unk,
            "compression": n_cr, "logprob": n_lp}


def _lib_version(mod: str) -> str:
    """버전 문자열. 못 읽으면 빈 문자열이 아니라 '모름'을 적는다 —
    빈 값은 '버전이 없음'으로 읽힌다."""
    try:
        import importlib
        return str(getattr(importlib.import_module(mod), "__version__", "모름"))
    except Exception:  # noqa: BLE001
        return "모름"


def _perf_cores(default: int = 6) -> int:
    """성능(P) 코어 수. 없으면 default.

    ctranslate2 는 cpu_threads 미지정 시 논리 코어 전체(M1 Pro 는 8)를 쓴다.
    그러면 효율(E) 코어에도 스레드가 걸리고, P 코어가 먼저 끝나 E 코어를
    기다리면서 동기화 대기가 생긴다.

    실측(2026-08-04, M1 Pro 6P+2E, large-v3-turbo int8, 100초 오디오):
        cpu_threads 기본  40.9초 (2.44배속)
        cpu_threads 6     17.7초 (5.64배속)   ← 2.3배 빠름
        cpu_threads 4     18.2초 (5.48배속)

    이 최적화는 원래 `fast` 분기에만 들어가 있었는데, 그 분기는 검증 실패로
    쓰지 않는 죽은 코드다. 정작 쓰는 경로에는 빠져 있었다.
    """
    global _PERF_CORES_SOURCE
    try:
        out = subprocess.run(["sysctl", "-n", "hw.perflevel0.logicalcpu"],
                             capture_output=True, text=True, timeout=2)
        n = int(out.stdout.strip())
        if n > 0:
            _PERF_CORES_SOURCE = "sysctl:hw.perflevel0.logicalcpu"
            return n
    except Exception:  # noqa: BLE001
        pass
    # sysctl 이 없거나 실패한 기계(맥이 아니거나 sysctl 키가 없음). 성능코어를
    # **못 잰 것**이지 성능코어가 default 개인 게 아니다. 출처를 남겨 둔다.
    _PERF_CORES_SOURCE = "fallback:min(default,os.cpu_count)"
    return min(default, os.cpu_count() or default)


def transcribe_file(audio_path: str, speaker: str,
                    model_size: str = "small",
                    language: Optional[str] = None,
                    device: str = "auto",
                    dual_pass: bool = False,
                    fast: bool = False,
                    multilingual: bool = False,
                    no_vad: bool = False,
                    # ── 아래 셋은 2026-08-24 추가 (빈칸·환각 A/B 실험용).
                    #    기본값이 전부 **현행 실효값과 같다** — 기존 76+세션 재현성이 안 깨진다.
                    #    condition_on_previous_text=True 는 faster-whisper 기본값이고,
                    #    hallucination_silence=None 이면 종전대로 no_vad 일 때만 2.0 이 걸린다.
                    condition_on_previous_text: bool = True,
                    hallucination_silence: Optional[float] = None,
                    vad_threshold: float = 0.5,
                    # 2026-08-25: 채우기 패스를 여러 겹으로. `_fill_gaps` 는 **침묵 구간에만**
                    # 끼워 넣으므로 아무리 공격적으로 걸어도 본문을 못 망친다.
                    # 근거: 이 코퍼스에서 VAD 가 GPT 음성(스피커 재생음 재수음)을 통째로
                    # 버린다(아래 _decode 주석 · 5.4배·4.1배 회복). 그래서 U→GPT 사이에
                    # 있을 수 없는 긴 공백이 남는다(사용자 전제 8/25: user 끝 → GPT 시작은 2초 이내).
                    fill_thresholds: Optional[List[float]] = None,
                    _model_cache={}) -> List[Word]:
    """Transcribe one audio file; every word is tagged with `speaker`.

    When diarization will run afterwards, pass speaker="" and relabel the
    returned words with gblite.diarize.assign_speakers().

    fast=True 는 **검증 실패했다. 쓰지 말 것.**
    BatchedInferencePipeline(batch_size=8)은 2.02배 빠르지만, ChatGPT 실험
    음성 10분 발췌에서 단어 995→705(-29%)로 무너졌다. 침묵 마커도 함께
    붕괴(pause 78→50, largepause 52→35). 배치 모드의 VAD 분절이 발화를
    통째로 버리기 때문이다. 재현: `python verify_fast.py <오디오>`.
    구현은 조사 기록으로 남겨두고 기본은 꺼져 있다.

    multilingual=True 는 **구간마다 언어를 다시 판단**한다. 기본(False)에서는
    faster-whisper 가 파일 앞 30초에서 언어를 한 번 정해 파일 전체에 적용하는데,
    이 코퍼스처럼 세션 앞부분에서 연구원이 영어로 절차를 안내하면 한국어 세션이
    통째로 영어로 고정된다. 그러면 한국어 발화가 영어 반복구로 붕괴한다.

    실측(참여자B 4:32, 같은 오디오):
        기본           "I'm going to talk about the ticket sales and management
                        business." / "Yes."           ← 내용 소실
        multilingual   "네, 저는 영화관에서 티켓 판매와 관객 안내 업무 담당
                        아르바이트에 지원했습니다…"    ← 복구

    language 를 지정하면 그 값이 우선이므로 multilingual 은 무시된다.
    한영 혼용 세션에는 language 고정보다 이쪽이 맞다 — 고정하면 반대 언어
    구간에서 같은 붕괴가 일어난다."""
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        sys.exit("faster-whisper가 설치되어 있지 않습니다. "
                 "먼저 `pip install faster-whisper` 를 실행하세요.")

    dev = "cpu" if device == "auto" else device
    key = (model_size, device, fast)
    # 캐시 적중이든 아니든 같은 값이다(_perf_cores 는 이 기계에서 결정적).
    # 그래서 모델을 새로 만들지 않는 경로에서도 기록할 수 있다.
    threads = _perf_cores()
    if key not in _model_cache:
        # int8 quantization: fast + low memory on Apple Silicon CPUs
        if fast:
            from faster_whisper import BatchedInferencePipeline
            # cpu_threads=6: M1 Pro 성능코어 수. 기본값에 맡기면 효율코어에
            # 스레드가 걸려 동기화 대기가 생긴다.
            base = WhisperModel(model_size, device=dev, compute_type="int8",
                                cpu_threads=threads)
            _model_cache[key] = BatchedInferencePipeline(model=base)
        else:
            _model_cache[key] = WhisperModel(model_size, device=dev,
                                             compute_type="int8",
                                             cpu_threads=threads)
    model = _model_cache[key]
    _RUN_CONFIG.update({
        "model_size": model_size,
        "device": dev,
        "compute_type": "int8",
        "cpu_threads": threads,
        "cpu_threads_source": _PERF_CORES_SOURCE,
        "cpu_count_logical": os.cpu_count(),
        "batched": bool(fast),          # --fast (BatchedInferencePipeline)
        "dual_pass": bool(dual_pass),
        "condition_on_previous_text": bool(condition_on_previous_text),
        "hallucination_silence": hallucination_silence,
        "vad_threshold": vad_threshold,
        "faster_whisper": _lib_version("faster_whisper"),
        "ctranslate2": _lib_version("ctranslate2"),
    })

    _cond = "1" if condition_on_previous_text else "0"
    _hal = hallucination_silence if hallucination_silence is not None else (2.0 if no_vad else None)
    _tag1 = ("p1-novad" if no_vad else f"p1-vad{vad_threshold:.2f}")
    words = _decode(model, audio_path, language, speaker,
                    threshold=vad_threshold, batched=fast, multilingual=multilingual,
                    no_vad=no_vad,
                    condition_on_previous_text=condition_on_previous_text,
                    hallucination_silence=hallucination_silence,
                    pass_tag=f"{_tag1}-cond{_cond}" + (f"-hal{_hal}" if _hal else ""))
    # 채우기 패스 목록을 정한다. --dual-pass 는 [0.35] 와 같다(기존 동작 보존).
    fills = list(fill_thresholds) if fill_thresholds else ([0.35] if dual_pass else [])
    for fi, ft in enumerate(fills, 1):
        # 임계값 하나로는 모든 발화를 못 잡는다(docs/챗봇_VAD_설정.md · 개발기록 2.1).
        # 0.5(A안)는 챗봇 TTS 의 평탄한 낭독을 통째로 놓친다.
        #
        # ⚠️ 자주 잘못 인용되는 것 (2026-08-24 정정): 발화를 잃은 건 0.35 가 아니라 B안이고,
        #    B안은 문턱만이 아니라 threshold 0.25 + speech_pad_ms 700 + min_silence 2000
        #    + condition_on_previous_text=False 의 **묶음**이다.
        #
        # `_fill_gaps` 는 1차가 침묵으로 남긴 구간에만 넣으므로 낮은 문턱을 겹쳐도 안전하다.
        # ft == 0 은 「VAD 를 통째로 끈 채우기」다 (2026-08-25 · 개발기록 2.98).
        # 재수음 GPT 긴 발화는 Silero 가 문턱 0.05 로도 못 본다(청취 11/11 확정) —
        # 문턱을 내리는 게 아니라 꺼야 그 소리가 whisper 에 닿는다.
        # no_vad 경로는 환각 가드(hallucination_silence 2.0)가 자동으로 걸리고,
        # 어차피 _fill_gaps 가 구멍에만 끼우므로 본문은 못 건드린다.
        빈칸채우기_novad = (ft == 0)
        low = _decode(model, audio_path, language, speaker,
                      threshold=ft, batched=fast, multilingual=multilingual,
                      no_vad=빈칸채우기_novad,
                      condition_on_previous_text=condition_on_previous_text,
                      hallucination_silence=hallucination_silence,
                      pass_tag=(f"p{fi+1}-novad-cond{_cond}" if 빈칸채우기_novad
                                else f"p{fi+1}-vad{ft:.2f}-cond{_cond}"))
        words = _fill_gaps(words, low)
    return words


def _decode(model, audio_path: str, language: Optional[str],
            speaker: str, threshold: float,
            batched: bool = False,
            multilingual: bool = False,
            no_vad: bool = False,
            condition_on_previous_text: bool = True,
            hallucination_silence: Optional[float] = None,
            pass_tag: str = "p1") -> List[Word]:
    # no_vad: Silero VAD 를 통째로 끈다.
    #
    # 이 코퍼스에서는 VAD 가 GPT 음성(스피커 재생음이 방 마이크로 재수음돼
    # 열화됨)을 비음성으로 판정해 발화를 통째로 버린다. 7세션 5분 구간 실측:
    # 참여자F 5.4배, 참여자C 4.1배 회복. 정상 세션(참여자E)은 1.0배로
    # 손해가 없다. 대신 느려지고 무음 구간 환각이 늘 수 있으므로
    # hallucination_silence_threshold 를 함께 건다.
    kwargs = dict(
        language=language,
        word_timestamps=True,
        vad_filter=not no_vad,
        # 2026-08-24: 여태 넘기지 않아 faster-whisper 기본값 True 가 그대로 먹었다
        # (5.2 (c) 에서 2026-08-13 에 지적하고 미반영). 이제 명시로 넘긴다.
        # True 를 명시하는 것은 생략과 동일하므로 기존 산출물과 바이트가 같다.
        condition_on_previous_text=condition_on_previous_text,
    )
    if not no_vad:
        kwargs["vad_parameters"] = {"threshold": threshold}
    # hallucination_silence_threshold 를 no_vad 밖으로 뺐다(2026-08-24).
    # 미지정이면 종전과 완전히 같다 — no_vad 면 2.0, VAD 켜면 안 걸림.
    # ⚠️ 이 옵션은 이상 세그먼트를 **삭제**한다(faster_whisper/transcribe.py:1332-1338).
    #    빈칸을 없애려다 빈칸을 만들 수 있으므로 켤 때는 반드시 구멍 수를 같이 잰다.
    _hal = hallucination_silence if hallucination_silence is not None else (2.0 if no_vad else None)
    if _hal is not None:
        kwargs["hallucination_silence_threshold"] = _hal
    # language 가 지정되면 그쪽이 우선이므로 굳이 켜지 않는다.
    if multilingual and language is None:
        kwargs["multilingual"] = True
    if batched:
        kwargs["batch_size"] = 8
    # 패스마다 VAD 설정이 다르다(dual_pass 는 0.50 뒤에 0.35 를 덧댄다).
    # 하나로 뭉뚱그리면 어느 조건으로 나온 낱말인지 나중에 못 가른다.
    _RUN_PASSES.append({
        "pass": pass_tag,
        "source": os.path.basename(audio_path),
        "vad_filter": bool(kwargs["vad_filter"]),
        "vad_threshold": (None if no_vad else threshold),
        "hallucination_silence_threshold":
            kwargs.get("hallucination_silence_threshold"),
        "batch_size": kwargs.get("batch_size"),
        "multilingual": bool(kwargs.get("multilingual", False)),
        "language": language,
        # ⚠️ **요청값이 아니라 실효값**을 적는다. BatchedInferencePipeline 은
        #    condition_on_previous_text=False · hallucination_silence_threshold=None 을
        #    하드코딩해 요청을 삼킨다(faster_whisper/transcribe.py:546-547).
        #    기록이 요청값을 적으면 거짓말이 된다.
        "condition_on_previous_text": (False if batched else condition_on_previous_text),
        "batched_ignored": (["condition_on_previous_text",
                             "hallucination_silence_threshold"] if batched else []),
    })
    segments, _ = model.transcribe(audio_path, **kwargs)
    words: List[Word] = []
    source = os.path.basename(audio_path)
    for seg in segments:
        # 낱말로 쪼개기 **전에** 세그먼트 신뢰도를 남긴다. 아래 낱말 루프는
        # 구두점만 남은 낱말을 버리는데, 그 버림이 세그먼트 층에 번지면
        # 안 된다.
        _record_segment(seg, source, pass_tag)
        for w in seg.words or []:
            # strip trailing punctuation the way GailBot's parsers do
            text = w.word.strip().strip(".,!?;:…")
            if not text:
                continue
            words.append(Word(start=float(w.start), end=float(w.end),
                              text=text, speaker=speaker,
                              provenance=pass_tag))
    return words


def _fill_gaps(primary: List[Word], secondary: List[Word],
               min_gap: float = 1.5) -> List[Word]:
    """primary가 침묵으로 남긴 구간에 secondary의 발화를 끼워 넣는다.

    구간(gap) 단위로만 삽입하므로 같은 말이 두 번 들어가지 않고,
    문장이 두 전사본 사이에서 쪼개지지도 않는다."""
    if not secondary:
        return primary
    primary = sorted(primary, key=lambda w: w.start)
    if not primary:
        return sorted(secondary, key=lambda w: w.start)

    gaps = []
    if primary[0].start >= min_gap:
        gaps.append((0.0, primary[0].start))
    for cur, nxt in zip(primary, primary[1:]):
        if nxt.start - cur.end >= min_gap:
            gaps.append((cur.end, nxt.start))
    # 마지막 구간은 실제 음원 길이를 모르는 상태의 열린 tail이다. 이
    # 구간에서 secondary를 채택하더라도 버리지 않되, 무음 환각 검토 대상임을
    # 낱말에 표시한다.
    tail_start = primary[-1].end
    gaps.append((tail_start, float("inf")))

    added = []
    for w in secondary:
        if any(gs <= w.start and w.end <= ge for gs, ge in gaps):
            # _decode가 만든 Word는 provenance를 갖지만, 외부 호출자가 만든
            # Word도 안전하게 보존한다. metadata는 새 객체에 복사한다.
            in_tail = w.start >= tail_start
            added.append(Word(start=w.start, end=w.end, text=w.text,
                              speaker=w.speaker, provenance=w.provenance,
                              tail_candidate=(w.tail_candidate or in_tail)))
    return sorted(primary + added, key=lambda w: w.start)


def speaker_name_for(path: str, index: int, n_files: int) -> str:
    """One audio file per speaker: use the file stem as the speaker name."""
    stem = os.path.splitext(os.path.basename(path))[0]
    return stem if n_files > 1 else "0"
