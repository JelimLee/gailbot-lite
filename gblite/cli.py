# -*- coding: utf-8 -*-
"""
GailBot-Lite CLI.

Usage examples
--------------
# 1) transcribe + CA annotation (one file per speaker):
python -m gblite transcribe speakerA.wav speakerB.wav -o results/

# 2) single mixed file (one speaker label '0'):
python -m gblite transcribe interview.mp3 -o results/ --language ko

# 3) annotation only, from existing GailBot word-level output:
python -m gblite annotate utt.toml -o results/
"""
import argparse
import os
import sys
import time

from . import align, analysis, loaders, output
from .transcribe import (transcribe_file, speaker_name_for,
                         reset_segments, collected_segments,
                         write_segments_csv, run_config, reset_run_config)


# 기본 문턱 프리셋. 2026-08-06 교수님 미팅에서 침묵 문턱 0.45초(measured)로
# 확정. 전에는 "gailbot"(영어 대면 대화용 0.1초)이 기본이라 한국어 화상
# 코퍼스에서 미세침묵이 6배 과다 탐지됐다 — 실측 근거는 analysis.py 의
# THRESHOLD_PRESETS 주석 참조.
DEFAULT_PRESET = "measured"


def _run_analysis_and_write(words_by_source, outdir, name, lang,
                            audio_path=None, extra_markers=None,
                            overlap_spans=None, preset=DEFAULT_PRESET):
    all_words = [w for ws in words_by_source.values() for w in ws]
    if not all_words:
        sys.exit("인식된 단어가 없습니다. 오디오/입력 파일을 확인해 주세요.")
    th = analysis.THRESHOLD_PRESETS.get(preset)
    if th is None:
        sys.exit(f"알 수 없는 문턱 프리셋: {preset} "
                 f"(가능: {', '.join(analysis.THRESHOLD_PRESETS)})")
    if preset != "gailbot":
        print(f"    문턱 프리셋 '{preset}' — 미세침묵 {th.LB_MICROPAUSE}~"
              f"{th.UB_MICROPAUSE}초 · 침묵 {th.LB_PAUSE}초~ · 간격 {th.GAPS_LB}초~")
    utts, markers, stats = analysis.analyze(all_words, th=th,
                                            audio_path=audio_path,
                                            overlap_spans=overlap_spans)
    if extra_markers:
        markers = sorted(markers + extra_markers, key=lambda m: (m.start, m.end))
    # 실행 조건을 산출물에 남긴다(2026-08-21). `cpu_threads` 는 기계마다 다른
    # 값이 되는데 지금까지 어디에도 안 적혔다 — 기계가 바뀌면 ctranslate2 의
    # 스레드 분할이 바뀌어 디코딩 결과가 갈릴 수 있고, 나중에 알아낼 방법이
    # 없다. **기존 키는 건드리지 않고 `run_config` 한 칸만 더한다**
    # (stats.json 소비자는 median/medianAbsDev 등을 이름으로 읽는다).
    # 전사를 안 거친 경로(annotate)에서는 run_config() 가 None 이라 칸 자체가
    # 안 생긴다 — 없는 조건을 지어내지 않는다.
    rc = run_config()
    if rc:
        stats = dict(stats or {})
        stats["run_config"] = rc
        print(f"    실행 조건 기록 — 모델 {rc['model_size']} · "
              f"cpu_threads {rc['cpu_threads']}({rc['cpu_threads_source']}) · "
              f"{rc['compute_type']} · batched {rc['batched']} · "
              f"패스 {len(rc['passes'])}개")
    paths = output.write_all(outdir, name, utts, markers,
                             words_by_source, stats, lang=lang)
    # 기존 *_words.csv의 4열 계약은 유지하고, 다중 패스 출처와 열린 tail
    # 후보는 별도 sidecar로 남긴다.
    if any(w.provenance or w.tail_candidate for w in all_words):
        meta_path = os.path.join(outdir, f"{name}_word_metadata.csv")
        output.write_csv_word_metadata(meta_path, all_words)
        paths.append(meta_path)
    # 세그먼트 신뢰도(Radford et al. 2022 §4.5)는 기존 산출물과 **별도 파일**로
    # 나간다. .cha/.txt/*_turns.csv/*_words.csv 형식은 건드리지 않는다.
    # annotate 처럼 전사를 안 거친 경로에서는 세그먼트가 없으므로 쓰지 않는다.
    segs = collected_segments()
    if segs:
        seg_path = os.path.join(outdir, f"{name}_segments.csv")
        write_segments_csv(seg_path, segs)
        paths.append(seg_path)
    reset_segments()
    reset_run_config()      # 세션 경계. 앞 세션 조건이 새어 들면 안 된다
    n_speakers = len({u.speaker for u in utts})
    print(f"\n완료: 발화 {len(utts)}개, 화자 {n_speakers}명, "
          f"CA 마커 {len(markers)}개")
    for p in paths:
        print(f"  → {p}")


def _maybe_align(words, audio, args):
    """강제 정렬을 켠 경우 낱말 시간을 다시 잡는다.

    전사·화자 분리 **다음**, 분석 **앞**에 와야 한다. 침묵·겹침 판정이
    타임스탬프를 읽으므로 순서가 뒤집히면 효과가 없다.
    """
    if not getattr(args, "align", False) or not audio:
        return words
    before = align.gap_summary(words)
    print(f"    강제 정렬 중 (낱말 {before['n_words']}개)...")
    words = align.align_words(words, audio)
    after = align.gap_summary(words)
    print(f"    정렬 완료 — 발화시간 {before['speech_sec']/60:.1f}분 "
          f"→ {after['speech_sec']/60:.1f}분 · "
          f"미세침묵대 간격 {before['micropause']} → {after['micropause']} · "
          f"침묵대 {before['pause']} → {after['pause']}")
    return words


def _fill_list(v):
    """--fill-thresholds "0.35,0.15" → [0.35, 0.15]. 미지정이면 None(= --dual-pass 기존 동작)."""
    if not v:
        return None
    return [float(x) for x in str(v).replace(" ", "").split(",") if x]


def cmd_transcribe(args):
    t0 = time.time()
    dropped = reset_segments()
    reset_run_config()
    if dropped:
        print(f"    [주의] 앞 실행의 세그먼트 {dropped}개를 비우고 시작합니다")
    diarize = getattr(args, "diarize", False)
    words_by_source = {}
    all_overlap_spans = []      # 화자 분리가 알려준 겹침 구간
    n = len(args.audio)
    for i, audio in enumerate(args.audio):
        if not os.path.exists(audio):
            sys.exit(f"파일을 찾을 수 없습니다: {audio}")
        name = os.path.splitext(os.path.basename(audio))[0]

        if diarize:
            # transcribe first (speaker unknown), then split by diarization
            print(f"[{i+1}/{n}] 전사 중: {audio} (모델 {args.model}, 화자 분리 ON)")
            words = transcribe_file(audio, "", model_size=args.model,
                                    language=args.language,
                                    multilingual=getattr(args, 'multilingual', False),
                                    no_vad=getattr(args, 'no_vad', False),
                                    condition_on_previous_text=not getattr(args, 'no_condition_prev', False),
                                    hallucination_silence=getattr(args, 'hallucination_silence', None),
                                    vad_threshold=getattr(args, 'vad_threshold', 0.5),
                                    fill_thresholds=_fill_list(getattr(args, 'fill_thresholds', None)))
            print(f"    단어 {len(words)}개 인식")
            from .diarize import (diarize_turns, assign_speakers,
                                  overlap_spans as _ov_spans, DiarizationError)
            try:
                turns = diarize_turns(
                    audio, hf_token=args.hf_token,
                    num_speakers=args.num_speakers,
                    min_speakers=args.min_speakers,
                    max_speakers=args.max_speakers)
            except DiarizationError as e:
                sys.exit(f"\n[화자 분리 오류]\n{e}")
            n_spk = len({t[2] for t in turns})
            print(f"    화자 {n_spk}명, 발화 구간 {len(turns)}개 감지")
            # prefix labels with file stem only when multiple files
            prefix = f"{name}_" if n > 1 else ""
            words = assign_speakers(words, turns, label_prefix=prefix)
            ov = _ov_spans(turns)
            if ov:
                print(f"    겹침 구간 {len(ov)}곳 "
                      f"({sum(e - s for s, e in ov):.1f}초)")
                all_overlap_spans.extend(ov)
        else:
            speaker = speaker_name_for(audio, i, n)
            print(f"[{i+1}/{n}] 전사 중: {audio} "
                  f"(화자 '{speaker}', 모델 {args.model})")
            words = transcribe_file(audio, speaker,
                                    model_size=args.model,
                                    language=args.language,
                                    multilingual=getattr(args, 'multilingual', False),
                                    no_vad=getattr(args, 'no_vad', False),
                                    condition_on_previous_text=not getattr(args, 'no_condition_prev', False),
                                    hallucination_silence=getattr(args, 'hallucination_silence', None),
                                    vad_threshold=getattr(args, 'vad_threshold', 0.5),
                                    fill_thresholds=_fill_list(getattr(args, 'fill_thresholds', None)))
            print(f"    단어 {len(words)}개 인식")
        words = _maybe_align(words, audio, args)
        words_by_source[name] = words

    # 운율 분석(억양·음량·호흡·웃음)을 켠다.
    #
    # 이전에는 audio_path 를 안 넘겨 transcribe 경로에서 운율이 통째로 꺼져
    # 있었다(batch 경로만 켜져 있었다). 사람 CA 전사본과 대조해 보니 사람은
    # 들숨(.hh)을 39곳, 웃음을 6곳 찍는데 게일봇은 **0곳**이었다 — 기능이
    # 없는 게 아니라 호출이 안 되고 있었다.
    #
    # 억양(↗↘)·작게(°)는 사람 전사본에 없는 표기지만 그대로 낸다. 기계가
    # 사람보다 잘하는 부분을 굳이 버릴 이유가 없다.
    #
    # 파일이 여럿이면(화자별 파일 입력) 어느 음원으로 운율을 잴지 정할 수
    # 없으므로 끈다 — analyze() 가 음원 하나만 받는다.
    _run_analysis_and_write(words_by_source, args.output,
                            args.name, args.chat_lang,
                            audio_path=(args.audio[0] if n == 1 else None),
                            overlap_spans=all_overlap_spans or None,
                            preset=getattr(args, 'thresholds', DEFAULT_PRESET))
    print(f"소요 시간: {time.time() - t0:.1f}초")


def _overlap_from_markers(path):
    """이전 산출물의 markers.csv 에서 겹침 구간을 되읽는다.

    `dz_overlap_start` / `dz_overlap_end` 가 시각 순으로 짝을 이룬다.
    개수가 안 맞으면 **조용히 넘기지 않고** 알린 뒤 None 을 낸다 —
    반쪽짜리 구간으로 표기를 만들면 어디가 틀렸는지 나중에 못 찾는다.
    """
    if not path:
        return None
    if not os.path.exists(path):
        sys.exit(f"겹침 원본을 찾을 수 없습니다: {path}")
    import csv as _csv
    st, en = [], []
    with open(path, encoding="utf-8-sig", newline="") as fh:
        for row in _csv.DictReader(fh):
            k = {x.lower(): x for x in row}
            t = row.get(k.get("type", "TYPE"), "")
            try:
                v = float(row[k["start time"]])
            except (KeyError, ValueError):
                continue
            if t == "dz_overlap_start":
                st.append(v)
            elif t == "dz_overlap_end":
                en.append(v)
    st.sort(); en.sort()
    if len(st) != len(en):
        print(f"    [경고] 겹침 시작 {len(st)}개 ≠ 끝 {len(en)}개 — "
              f"겹침을 싣지 않습니다: {path}")
        return None
    spans = [(s, e) for s, e in zip(st, en) if e > s]
    print(f"    겹침 구간 {len(spans)}곳을 이전 산출물에서 되읽었습니다")
    return spans or None


def cmd_annotate(args):
    if not os.path.exists(args.input):
        sys.exit(f"파일을 찾을 수 없습니다: {args.input}")
    words_by_source = loaders.load(args.input)
    total = sum(len(v) for v in words_by_source.values())
    print(f"불러오기 완료: 소스 {len(words_by_source)}개, 단어 {total}개")
    # --audio 를 주면 원본 음원으로 운율까지 다시 분석한다.
    #
    # 이게 필요한 이유: 억양(↗↘, 말미 구두점)·음량·호흡·웃음·강조는 전부
    # 파형에서 나온다. utt.toml 에는 낱말과 시각만 있어, 음원 없이 재주석을
    # 돌리면 이 마커들이 통째로 빠진 전사본이 나온다. 전에는 annotate 에
    # 음원을 줄 방법 자체가 없어, 문턱만 바꿔 다시 돌리면 운율 표기를
    # 잃어버렸다 (2026-08-06).
    audio = getattr(args, "audio", None)
    if audio and not os.path.exists(audio):
        sys.exit(f"오디오 파일을 찾을 수 없습니다: {audio}")

    # 겹침은 화자 분리에서 나오는데 utt.toml 에는 실리지 않는다. 그래서
    # annotate 만 다시 돌리면 겹침 표기와 겹침 기반 절단(`cutoff,overlap`)이
    # **조용히 사라진다** — 2026-08-14 에 8세션에서 실제로 밟았다
    # (겹침 94→0, 절단 48→3). 이전 산출물의 markers.csv 에서 구간을 되읽어
    # 그 손실을 막는다. 같은 배선 누락이 realign_batch.py 에도 있었다.
    ov = _overlap_from_markers(getattr(args, "overlap_from", None))
    _run_analysis_and_write(words_by_source, args.output,
                            args.name, args.chat_lang,
                            audio_path=audio, overlap_spans=ov,
                            preset=getattr(args, 'thresholds', DEFAULT_PRESET))


# 영상 컨테이너도 넣는다. 오디오는 PyAV 로 디코딩하므로 컨테이너 종류를
# 가리지 않는다. 사람 간 세션의 원본은 캠코더 MTS(AVCHD)이고, 별도로 mp3 를
# 추출해 두지 않은 녹화가 있어 원본을 그대로 받을 수 있어야 한다(2026-08-11).
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".mp4", ".flac", ".aac",
              ".ogg", ".opus", ".wma", ".aiff", ".aif",
              ".mts", ".m2ts", ".mkv", ".mov", ".avi", ".wmv"}


def _find_audio_files(folder: str):
    files = []
    for entry in sorted(os.listdir(folder)):
        path = os.path.join(folder, entry)
        if os.path.isfile(path) and os.path.splitext(entry)[1].lower() in AUDIO_EXTS:
            files.append(path)
    return files


def cmd_batch(args):
    """Process every audio file in a folder INDEPENDENTLY.

    Each file is treated as its own recording and produces its own set of
    output files under <output>/<filename>/. This is the right mode for a
    folder of separate recordings (e.g. many interview sessions)."""
    t0 = time.time()
    if not os.path.isdir(args.folder):
        sys.exit(f"폴더를 찾을 수 없습니다: {args.folder}")
    files = _find_audio_files(args.folder)
    if not files:
        sys.exit(f"폴더에 오디오 파일이 없습니다: {args.folder}\n"
                 f"(지원 형식: {', '.join(sorted(AUDIO_EXTS))})")

    diarize = getattr(args, "diarize", False)
    print(f"총 {len(files)}개 파일 처리 시작 "
          f"(모델 {args.model}{', 화자 분리 ON' if diarize else ''})\n")

    ok, failed, skipped = 0, [], 0
    for idx, audio in enumerate(files, 1):
        name = os.path.splitext(os.path.basename(audio))[0]
        out_sub = os.path.join(args.output, name)
        print(f"═══ [{idx}/{len(files)}] {os.path.basename(audio)} ═══")
        # 앞 파일이 분석 전에 실패했다면 그 세그먼트가 남아 있다. 남은 채로
        # 이어 쓰면 다음 세션 CSV 에 조용히 섞이므로, 몇 개를 버리는지 찍고
        # 비운다.
        dropped = reset_segments()
        reset_run_config()
        if dropped:
            print(f"    [주의] 직전 파일의 미기록 세그먼트 {dropped}개 폐기")
        if getattr(args, "skip_done", False) and \
                os.path.exists(os.path.join(out_sub, "utt.toml")):
            print("    [건너뜀] 이미 처리된 결과가 있습니다\n")
            skipped += 1
            continue
        try:
            extra = None
            if getattr(args, "multichannel", False):
                from .multichannel import (split_channels, channels_are_distinct,
                                           suppress_crosstalk, merge_words,
                                           detect_true_overlaps)
                chans = split_channels(audio)
                if len(chans) < 2 or not channels_are_distinct(chans):
                    print("    [경고] 채널이 분리돼 있지 않습니다(dual-mono). "
                          "겹침을 잡을 수 없어 일반 처리로 진행합니다.")
                else:
                    per = {}
                    for ci, cp in enumerate(chans):
                        cw = transcribe_file(cp, f"CH{ci+1}",
                                             model_size=args.model,
                                             language=args.language,
                                             dual_pass=args.dual_pass,
                                             multilingual=getattr(args, 'multilingual', False),
                                             no_vad=getattr(args, 'no_vad', False),
                                             condition_on_previous_text=not getattr(args, 'no_condition_prev', False),
                                             hallucination_silence=getattr(args, 'hallucination_silence', None),
                                             vad_threshold=getattr(args, 'vad_threshold', 0.5),
                                             fill_thresholds=_fill_list(getattr(args, 'fill_thresholds', None)))
                        per[f"CH{ci+1}"] = cw
                        print(f"    채널 {ci+1}: 단어 {len(cw)}개")
                    per = suppress_crosstalk(per)
                    extra = detect_true_overlaps(per)
                    words = merge_words(per)
                    print(f"    실제 겹침 {len(extra)//4}건 검출")
                    # 다채널 경로도 --thresholds 를 따르게 한다. 전에는
                    # 이 분기만 인자를 빼먹어 같은 배치 안에서 채널 분리
                    # 파일만 다른 문턱으로 분석됐다.
                    _run_analysis_and_write({name: words}, out_sub, args.name,
                                            args.chat_lang, audio_path=audio,
                                            extra_markers=extra,
                                            preset=getattr(args, 'thresholds',
                                                           DEFAULT_PRESET))
                    ok += 1
                    print()
                    continue
            fast = getattr(args, "fast", False)
            par = getattr(args, "parallel_diarize", False)
            if diarize:
                from .diarize import (diarize_turns, assign_speakers,
                                      overlap_spans as _ov_spans,
                                      DiarizationError)
                dkw = dict(hf_token=args.hf_token,
                           num_speakers=args.num_speakers,
                           min_speakers=args.min_speakers,
                           max_speakers=args.max_speakers)
                if par:
                    # Whisper는 CTranslate2(CPU 전용), pyannote는 MPS(GPU).
                    # 연산 장치가 달라 동시에 돌리면 실제로 겹친다.
                    # 실측(10분 발췌): 전사 180초 + 화자분리 48초 = 순차 228초
                    # → 병렬 시 상한 180초 (약 21% 단축).
                    # 두 함수는 같은 파일을 각자 읽고 공유 상태가 없으므로
                    # 결과는 순차 실행과 동일하다. 합치는 assign_speakers도
                    # 순서 무관.
                    from concurrent.futures import ThreadPoolExecutor
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        fut = pool.submit(diarize_turns, audio, **dkw)
                        words = transcribe_file(
                            audio, "", model_size=args.model,
                            language=args.language,
                            dual_pass=args.dual_pass, fast=fast,
                            multilingual=getattr(args, 'multilingual', False),
                            no_vad=getattr(args, 'no_vad', False),
                            condition_on_previous_text=not getattr(args, 'no_condition_prev', False),
                            hallucination_silence=getattr(args, 'hallucination_silence', None),
                            vad_threshold=getattr(args, 'vad_threshold', 0.5),
                            fill_thresholds=_fill_list(getattr(args, 'fill_thresholds', None)))
                        turns = fut.result()
                else:
                    words = transcribe_file(audio, "", model_size=args.model,
                                            language=args.language,
                                            dual_pass=args.dual_pass,
                                            fast=fast,
                                            multilingual=getattr(args, 'multilingual', False),
                                            no_vad=getattr(args, 'no_vad', False),
                                            condition_on_previous_text=not getattr(args, 'no_condition_prev', False),
                                            hallucination_silence=getattr(args, 'hallucination_silence', None),
                                            vad_threshold=getattr(args, 'vad_threshold', 0.5),
                                            fill_thresholds=_fill_list(getattr(args, 'fill_thresholds', None)))
                    turns = diarize_turns(audio, **dkw)
                print(f"    단어 {len(words)}개 인식")
                n_spk = len({t[2] for t in turns})
                print(f"    화자 {n_spk}명 감지")
                words = assign_speakers(words, turns)
                ov = _ov_spans(turns)
                if ov:
                    print(f"    겹침 구간 {len(ov)}곳 "
                          f"({sum(e - s for s, e in ov):.1f}초)")
            else:
                ov = None
                words = transcribe_file(audio, "0", model_size=args.model,
                                        language=args.language,
                                        dual_pass=args.dual_pass, fast=fast,
                                        multilingual=getattr(args, 'multilingual', False),
                                        no_vad=getattr(args, 'no_vad', False),
                                        condition_on_previous_text=not getattr(args, 'no_condition_prev', False),
                                        hallucination_silence=getattr(args, 'hallucination_silence', None),
                                        vad_threshold=getattr(args, 'vad_threshold', 0.5),
                                        fill_thresholds=_fill_list(getattr(args, 'fill_thresholds', None)))
                print(f"    단어 {len(words)}개 인식")
            words = _maybe_align(words, audio, args)
            _run_analysis_and_write({name: words}, out_sub,
                                    args.name, args.chat_lang,
                                    audio_path=audio, overlap_spans=ov,
                                    preset=getattr(args, 'thresholds', DEFAULT_PRESET))
            ok += 1
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001
            print(f"    [건너뜀] 오류: {e}")
            failed.append((name, str(e)))
        print()

    print("=" * 50)
    print(f"완료: 성공 {ok}개 / 실패 {len(failed)}개 "
          f"{f'/ 건너뜀 {skipped}개 ' if skipped else ''}"
          f"(총 {len(files)}개), 소요 {time.time()-t0:.1f}초")
    print(f"결과 폴더: {args.output}/<파일이름>/")
    if failed:
        print("\n실패한 파일:")
        for name, err in failed:
            print(f"  - {name}: {err}")


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="gblite",
        description="GailBot-Lite: Whisper 기반 대화분석(CA) 전사 파이프라인 "
                    "(macOS Apple Silicon용 재구현)")
    sub = p.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe",
                       help="오디오 전사 + CA 주석 (파일 1개 = 화자 1명)")
    t.add_argument("audio", nargs="+",
                   help="오디오 파일(wav/mp3/m4a/mp4 등). 여러 개 주면 "
                        "각 파일을 별도 화자로 처리")
    t.add_argument("-o", "--output", default="gblite_results",
                   help="결과 디렉토리 (기본: gblite_results)")
    t.add_argument("-m", "--model", default="small",
                   choices=["tiny", "base", "small", "medium", "large-v3",
                            "large-v3-turbo"],
                   help="Whisper 모델 크기 (기본: small)")
    t.add_argument("-l", "--language", default=None,
                   help="언어 코드 (예: en, ko). 생략하면 자동 감지")
    t.add_argument("--name", default="conversation",
                   help="출력 파일 이름 접두어 (기본: conversation)")
    t.add_argument("--chat-lang", default=None,
                   help="CHAT 헤더 언어 코드 (기본: eng)")
    # --- diarization (선택) ---
    t.add_argument("--diarize", action="store_true",
                   help="화자가 섞인 단일 녹음을 자동으로 화자별로 분리 "
                        "(pyannote 필요: bash install_mac.sh --diarize)")
    t.add_argument("--num-speakers", type=int, default=None,
                   help="화자 수를 알면 지정 (정확도 향상). 예: --num-speakers 2")
    t.add_argument("--min-speakers", type=int, default=None,
                   help="최소 화자 수 (--num-speakers 없이 범위만 줄 때)")
    t.add_argument("--max-speakers", type=int, default=None,
                   help="최대 화자 수")
    t.add_argument("--no-vad", action="store_true",
                   help="Silero VAD 를 끈다. 스피커 재생음처럼 열화된 음성을 "
                        "VAD 가 비음성으로 판정해 버리는 녹음에 쓴다 "
                        "(실측: 최대 5.4배 회복. 정상 녹음도 WER 2~4점 개선 — "
                        "2.27절. 선별 배율은 양만 재고 질은 못 잰다). "
                        "느려지고 무음 구간 환각이 늘 수 있다")
    t.add_argument("--no-condition-prev", action="store_true",
                     help="condition_on_previous_text=False. 직전 창의 전사문을 다음 창 "
                          "프롬프트로 넘기지 않는다 — 반복 환각 루프의 직접 처방. "
                          "(2026-08-24 A/B 실험용. 기본은 faster-whisper 기본값 True)")
    t.add_argument("--fill-thresholds", metavar="F1,F2", default=None,
                     help="채우기 패스 문턱들을 쉼표로. 예 `0.35,0.15`. "
                          "1차(--vad-threshold)가 침묵으로 남긴 구간에만 끼워 넣으므로 "
                          "낮게 걸어도 본문은 안 바뀐다. 미지정이면 --dual-pass 가 0.35 하나를 쓴다. "
                          "근거: 이 코퍼스는 VAD 가 GPT 재수음을 버려 U→GPT 사이에 "
                          "있을 수 없는 공백이 남는다(개발기록 2.96)")
    t.add_argument("--vad-threshold", type=float, default=0.5, metavar="F",
                     help="Silero VAD 문턱 (기본 0.5). 0.35 로 내리면 빈칸이 줄 수 있다 — "
                          "빈칸 원인의 54.9%%가 「VAD 가 말소리를 버림」이다(개발기록 2.86)")
    t.add_argument("--hallucination-silence", type=float, default=None, metavar="SEC",
                     help="hallucination_silence_threshold. 미지정이면 --no-vad 일 때만 2.0. "
                          "⚠️ 이상 세그먼트를 삭제하므로 빈칸이 늘 수 있다")
    t.add_argument("--thresholds", default=DEFAULT_PRESET,
                   choices=["gailbot", "gao2025", "measured", "aligned"],
                   help="침묵 문턱 프리셋. gailbot=원 논문 기본값(영어 대면 대화) · "
                        "gao2025=L2 화자 지각 문턱 350ms (Language Testing 2025) · "
                        "measured=우리 사람 CA 전사본 6건에 맞춘 값(기본). "
                        "gailbot 의 0.1초는 한국어 화상 대화에서 실측상 "
                        "6배 과다 탐지 — 2026-08-06 교수님 확정으로 기본값을 "
                        "measured(0.45초)로 옮겼다")
    t.add_argument("--align", action="store_true",
                   help="wav2vec2 강제 정렬로 낱말 시간을 다시 잡는다. "
                        "전사는 그대로 두고 시간만 고친다 — Whisper 가 "
                        "낱말을 늘여 잡아 침묵을 삼키는 것을 바로잡는다 "
                        "(실측: 미세침묵 28 → 514). 한/영 세그먼트를 "
                        "갈라 각각의 모델로 태운다. whisperx 필요")
    t.add_argument("--multilingual", action="store_true",
                   help="구간마다 언어를 다시 판단한다. 기본은 파일 앞 30초에서 "
                        "언어를 한 번 정해 전체에 적용하는데, 앞부분이 다른 "
                        "언어면(예: 영어 절차 안내) 본문이 통째로 그 언어로 "
                        "고정돼 내용이 붕괴한다. 한영 혼용 녹음에 켤 것. "
                        "-l 을 지정하면 무시된다")
    t.add_argument("--hf-token", default=None,
                   help="HuggingFace 토큰 (환경변수 HF_TOKEN 으로도 지정 가능)")
    t.set_defaults(func=cmd_transcribe)

    # --- batch: 폴더 전체 일괄 처리 (파일마다 독립 처리) ---
    b = sub.add_parser("batch",
                       help="폴더 안 모든 오디오를 각각 독립적으로 전사 "
                            "+ CA 주석 (파일별 결과 폴더 생성)")
    b.add_argument("folder", help="오디오 파일들이 든 폴더")
    b.add_argument("-o", "--output", default="gblite_results",
                   help="결과 상위 폴더 (기본: gblite_results). "
                        "파일별로 하위 폴더가 생김")
    b.add_argument("-m", "--model", default="small",
                   choices=["tiny", "base", "small", "medium", "large-v3",
                            "large-v3-turbo"],
                   help="Whisper 모델 크기 (기본: small)")
    b.add_argument("-l", "--language", default=None,
                   help="언어 코드 (예: en, ko). 생략하면 자동 감지")
    b.add_argument("--name", default="conversation",
                   help="각 결과 파일 이름 접두어 (기본: conversation)")
    b.add_argument("--chat-lang", default=None,
                   help="CHAT(.cha) 헤더 언어. 미지정 시 -l 값에서 자동 도출 "
                        "(ko→kor, 그 외→eng)")
    b.add_argument("--diarize", action="store_true",
                   help="각 파일마다 화자 자동 분리 (pyannote 필요)")
    b.add_argument("--num-speakers", type=int, default=None,
                   help="모든 파일의 화자 수가 같을 때 지정")
    b.add_argument("--min-speakers", type=int, default=None)
    b.add_argument("--max-speakers", type=int, default=None)
    b.add_argument("--hf-token", default=None,
                   help="HuggingFace 토큰 (환경변수 HF_TOKEN 으로도 지정 가능)")
    b.add_argument("--multichannel", action="store_true",
                   help="화자별로 채널이 분리된 녹음. 채널마다 따로 전사해 "
                        "**실제 동시 발화(겹침)**를 검출한다. dual-mono면 "
                        "경고 후 일반 처리로 넘어간다")
    b.add_argument("--dual-pass", action="store_true",
                   help="VAD 임계값 0.5/0.35로 두 번 전사해 누락 발화를 보완 "
                        "(2배 느리지만 챗봇 낭독조 드롭을 복구)")
    b.add_argument("--no-vad", action="store_true",
                   help="Silero VAD 를 끈다. 스피커 재생음처럼 열화된 음성을 "
                        "VAD 가 비음성으로 판정해 버리는 녹음에 쓴다 "
                        "(실측: 최대 5.4배 회복. 정상 녹음도 WER 2~4점 개선 — "
                        "2.27절. 선별 배율은 양만 재고 질은 못 잰다). "
                        "느려지고 무음 구간 환각이 늘 수 있다")
    b.add_argument("--no-condition-prev", action="store_true",
                     help="condition_on_previous_text=False. 직전 창의 전사문을 다음 창 "
                          "프롬프트로 넘기지 않는다 — 반복 환각 루프의 직접 처방. "
                          "(2026-08-24 A/B 실험용. 기본은 faster-whisper 기본값 True)")
    b.add_argument("--fill-thresholds", metavar="F1,F2", default=None,
                     help="채우기 패스 문턱들을 쉼표로. 예 `0.35,0.15`. "
                          "1차(--vad-threshold)가 침묵으로 남긴 구간에만 끼워 넣으므로 "
                          "낮게 걸어도 본문은 안 바뀐다. 미지정이면 --dual-pass 가 0.35 하나를 쓴다. "
                          "근거: 이 코퍼스는 VAD 가 GPT 재수음을 버려 U→GPT 사이에 "
                          "있을 수 없는 공백이 남는다(개발기록 2.96)")
    b.add_argument("--vad-threshold", type=float, default=0.5, metavar="F",
                     help="Silero VAD 문턱 (기본 0.5). 0.35 로 내리면 빈칸이 줄 수 있다 — "
                          "빈칸 원인의 54.9%%가 「VAD 가 말소리를 버림」이다(개발기록 2.86)")
    b.add_argument("--hallucination-silence", type=float, default=None, metavar="SEC",
                     help="hallucination_silence_threshold. 미지정이면 --no-vad 일 때만 2.0. "
                          "⚠️ 이상 세그먼트를 삭제하므로 빈칸이 늘 수 있다")
    b.add_argument("--thresholds", default=DEFAULT_PRESET,
                   choices=["gailbot", "gao2025", "measured", "aligned"],
                   help="침묵 문턱 프리셋. gailbot=원 논문 기본값(영어 대면 대화) · "
                        "gao2025=L2 화자 지각 문턱 350ms (Language Testing 2025) · "
                        "measured=우리 사람 CA 전사본 6건에 맞춘 값(기본). "
                        "gailbot 의 0.1초는 한국어 화상 대화에서 실측상 "
                        "6배 과다 탐지 — 2026-08-06 교수님 확정으로 기본값을 "
                        "measured(0.45초)로 옮겼다")
    b.add_argument("--align", action="store_true",
                   help="wav2vec2 강제 정렬로 낱말 시간을 다시 잡는다. "
                        "전사는 그대로 두고 시간만 고친다 — Whisper 가 "
                        "낱말을 늘여 잡아 침묵을 삼키는 것을 바로잡는다 "
                        "(실측: 미세침묵 28 → 514). 한/영 세그먼트를 "
                        "갈라 각각의 모델로 태운다. whisperx 필요")
    b.add_argument("--multilingual", action="store_true",
                   help="구간마다 언어를 다시 판단 (한영 혼용 녹음용). "
                        "-l 을 지정하면 무시된다")
    b.add_argument("--skip-done", action="store_true",
                   help="이미 결과가 있는 파일은 건너뜀 (중단된 작업 이어하기)")
    b.add_argument("--parallel-diarize", action="store_true",
                   help="[검증 실패 — 쓰지 말 것] 전사(CPU)와 화자분리(MPS) "
                        "동시 실행. 27%% 빠르지만 CPU 경합이 CTranslate2의 "
                        "부동소수점 축약 순서를 바꿔 전사 결과가 달라진다 "
                        "(843→884 단어). 조사 기록용")
    b.add_argument("--fast", action="store_true",
                   help="[검증 실패 — 쓰지 말 것] 배치 추론(batch_size=8). "
                        "2.02배 빠르지만 ChatGPT 실험 데이터에서 단어 29%%를 "
                        "잃었다(995→705). 조사 기록용으로만 남겨둠")
    b.set_defaults(func=cmd_batch)

    a = sub.add_parser("annotate",
                       help="이미 전사된 파일(utt.toml/csv)에 CA 주석만 적용")
    a.add_argument("input", help="utt.toml 또는 word-level CSV")
    a.add_argument("-o", "--output", default="gblite_results")
    a.add_argument("--name", default="conversation")
    a.add_argument("--chat-lang", default=None)
    a.add_argument("--audio", default=None,
                   help="원본 음원. 주면 운율(억양·음량·호흡·웃음·강조)까지 "
                        "다시 분석한다. 없으면 시간값으로 낼 수 있는 마커만 "
                        "생성된다")
    a.add_argument("--overlap-from", default=None, metavar="MARKERS.CSV",
                   help="이전 산출물의 conversation_markers.csv. 겹침 구간을 "
                        "여기서 읽어 온다. utt.toml 에는 겹침이 실리지 않아, "
                        "주지 않으면 겹침 표기와 겹침 기반 절단이 사라진다")
    a.add_argument("--thresholds", default=DEFAULT_PRESET,
                   choices=["gailbot", "gao2025", "measured", "aligned"],
                   help="침묵 문턱 프리셋 (기본: measured). "
                        "transcribe/batch 와 같은 값을 쓴다")
    a.set_defaults(func=cmd_annotate)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
