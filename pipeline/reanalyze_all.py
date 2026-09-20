# -*- coding: utf-8 -*-
"""이미 전사된 결과에 새 CA 마커(래칭 =, 절단 -, 반복)를 다시 입힌다.

전사(느림)는 건드리지 않고 분석 단계만 다시 돌리므로 몇 초면 끝난다.
배치가 도는 중에 코드를 고쳤기 때문에, 먼저 처리된 파일들에는 새 마커가
빠져 있다 — 이 스크립트로 전체를 같은 기준으로 맞춘다.
"""
import subprocess
import sys
from pathlib import Path

BASES = [
    Path(f"{Path.home()}/Downloads/01_게일봇연구/gailbot_results_v2"),
    Path(f"{Path.home()}/Downloads/01_게일봇연구/gailbot_results_ai_v2"),
    Path(f"{Path.home()}/Downloads/01_게일봇연구/gailbot_results_gpt2025"),
    Path(f"{Path.home()}/Downloads/01_게일봇연구/gailbot_results_gpt2026"),
]
PY = str(Path(__file__).parent / ".venv/bin/python")


def main():
    done = failed = 0
    for base in BASES:
        if not base.exists():
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            csv = folder / "conversation_words.csv"
            if not csv.exists():
                continue
            r = subprocess.run(
                [PY, "-m", "gblite", "annotate", str(csv), "-o", str(folder)],
                capture_output=True, text=True)
            if r.returncode == 0:
                done += 1
            else:
                failed += 1
                print(f"  실패: {folder.name} — {r.stderr.strip()[:80]}")
    print(f"재분석 완료: {done}개" + (f" / 실패 {failed}개" if failed else ""))


if __name__ == "__main__":
    sys.exit(main())
