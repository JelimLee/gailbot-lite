#!/bin/bash
# ============================================================
# GailBot-Lite 설치 스크립트 (macOS Apple Silicon)
# 사용법:  cd gailbot-lite && bash install_mac.sh
# ============================================================
set -e

echo "── Python 확인 ──"
if ! command -v python3 &>/dev/null; then
  echo "python3가 없습니다. https://www.python.org 또는 'brew install python' 후 다시 실행하세요."
  exit 1
fi
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
echo "python3 = $PYVER"
python3 - <<'EOF'
import sys
if sys.version_info < (3, 9):
    sys.exit("Python 3.9 이상이 필요합니다. brew install python 으로 최신 버전을 설치하세요.")
EOF

echo "── 가상환경 생성 (.venv) ──"
python3 -m venv .venv
source .venv/bin/activate

echo "── 패키지 설치 ──"
pip install --upgrade pip -q
pip install -r requirements.txt

# 운율 분석 — 억양(↑ ↓)·음량(° °)·호흡. CA 연구용이면 사실상 필수다.
# 없으면 그 마커들이 조용히 빠져 전사본 표기 체계가 갈린다.
echo "── 운율 분석 패키지 설치 (parselmouth) ──"
pip install -r requirements-prosody.txt
python -c "import parselmouth; print('parselmouth OK')" \
  || echo "  [경고] parselmouth 설치 실패 — 억양·음량 마커 없이 동작합니다"

# 선택: 화자 분리(diarization) 추가 설치  →  bash install_mac.sh --diarize
if [ "$1" == "--diarize" ]; then
  echo "── 화자 분리 패키지 설치 (pyannote + torch) ──"
  pip install -r requirements-diarize.txt
  python -c "import pyannote.audio; print('pyannote OK')"
  echo ""
  echo "  [화자 분리 사용 전 1회 설정]"
  echo "  1) https://huggingface.co/settings/tokens 에서 무료 토큰 발급"
  echo "  2) 아래 두 모델 페이지에서 'Agree' 클릭 (약관 동의):"
  echo "     - https://huggingface.co/pyannote/speaker-diarization-3.1"
  echo "     - https://huggingface.co/pyannote/segmentation-3.0"
  echo "  3) 실행 시  --hf-token <토큰>  또는  export HF_TOKEN=<토큰>"
fi

echo "── 설치 확인 ──"
python -c "import faster_whisper; print('faster-whisper OK:', faster_whisper.__version__)"
python -m gblite --help >/dev/null && echo "gblite CLI OK"

echo ""
echo "============================================================"
echo " 설치 완료!  사용 전 매번 가상환경을 켜세요:"
echo "   source .venv/bin/activate"
echo ""
echo " 사용 예:"
echo "   python -m gblite transcribe 오디오.wav -o results/"
echo "   python -m gblite annotate utt.toml -o results/"
echo "   python -m gblite transcribe 혼합녹음.mp3 --diarize --num-speakers 2 -o results/"
echo "============================================================"
