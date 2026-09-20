# -*- coding: utf-8 -*-
"""
Loaders for already-transcribed data (annotate-only mode).

Supports:
  - utt.toml  : GailBot word-level TOML ([[source]] tables)
  - .csv      : columns SPEAKER LABEL / TEXT / START TIME / END TIME
                (or lowercase speaker/text/start/end)
"""
import csv
import os
from typing import Dict, List

try:
    import tomllib  # Python >= 3.11
except ImportError:            # pragma: no cover
    import tomli as tomllib    # pip install tomli on 3.8-3.10

from .models import Word


def _toml_bool(value) -> bool:
    """Parse optional metadata without treating the string ``"0"`` as true."""
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def load_utt_toml(path: str) -> Dict[str, List[Word]]:
    """Returns {source_name: [Word, ...]}.

    If several sources share the same in-file speaker label (common when
    each speaker was recorded on a separate channel/file), the source
    name is used as the speaker so the conversation can be merged.
    """
    with open(path, "rb") as f:
        data = tomllib.load(f)

    sources = {name: rows for name, rows in data.items()
               if isinstance(rows, list)}

    # decide speaker labels
    speaker_sets = {name: {str(r.get("speaker", "")) for r in rows}
                    for name, rows in sources.items()}
    all_labels = [lb for s in speaker_sets.values() for lb in s]
    duplicated = len(sources) > 1 and len(set(all_labels)) < len(all_labels)

    out: Dict[str, List[Word]] = {}
    for name, rows in sources.items():
        words = []
        for r in rows:
            speaker = name if duplicated else str(r.get("speaker", name))
            words.append(Word(start=float(r["start"]), end=float(r["end"]),
                              text=str(r["text"]), speaker=speaker,
                              provenance=str(r.get("provenance", "")),
                              tail_candidate=_toml_bool(
                                  r.get("tail_candidate", False))))
        out[name] = words
    return out


def load_csv(path: str) -> Dict[str, List[Word]]:
    name = os.path.splitext(os.path.basename(path))[0]
    words: List[Word] = []
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        cols = {c.lower().replace(" ", ""): c for c in reader.fieldnames or []}

        def col(*names):
            for n in names:
                if n in cols:
                    return cols[n]
            raise SystemExit(f"CSV에 필요한 열이 없습니다: {names} / "
                             f"발견된 열: {reader.fieldnames}")

        c_sp = col("speakerlabel", "speaker")
        c_tx = col("text", "word")
        c_st = col("starttime", "start")
        c_en = col("endtime", "end")
        for row in reader:
            words.append(Word(start=float(row[c_st]), end=float(row[c_en]),
                              text=row[c_tx], speaker=str(row[c_sp]),
                              provenance=row.get("PROVENANCE", ""),
                              tail_candidate=_toml_bool(
                                  row.get("TAIL CANDIDATE", "0"))))
    return {name: words}


def load(path: str) -> Dict[str, List[Word]]:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".toml":
        return load_utt_toml(path)
    if ext == ".csv":
        return load_csv(path)
    raise SystemExit(f"지원하지 않는 입력 형식입니다: {ext} (toml/csv만 가능)")
