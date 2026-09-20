# GailBot-Lite

원본 [GailBot](https://github.com/mumair01/GailBot)(Tufts Human Interaction Lab)을
**macOS(Apple Silicon) 최신 환경에 맞게 새로 구현**한 대화분석(CA) 전사 파이프라인입니다.

원본은 2023년 기준으로 버전이 고정된 의존성(torch 1.13, tensorflow, pyannote 등 240여 개)
때문에 최신 macOS/Python에서 설치가 깨집니다. 이 재구현은 **의존성이 단 1개**
(faster-whisper)이며, ffmpeg나 conda 없이 동작합니다.

## 무엇을 하나요?

```
오디오 파일 → Whisper 음성인식(단어 단위 타임스탬프)
           → CA 주석 (pause / gap / overlap / 말속도)
           → 결과물 (CHAT .cha, TXT, CSV, utt.toml)
```

CA 주석 로직과 기준값은 원본 GailBot HiLabSuite 플러그인과 동일합니다:

| 현상 | 조건 (원본과 동일) | 표기 |
|---|---|---|
| Micropause | 같은 화자, 0.1–0.2초 침묵 | `(.)` |
| Pause | 같은 화자, 0.2–1.0초 | `(0.25)` |
| Large pause | 같은 화자, 1.0초 이상 | `(1.7)` |
| Latch | 같은 화자, 0.01–0.09초 | `≈` |
| Gap | 다른 화자 사이 0.3초 이상 침묵 | 별도 줄 `(1.2)` |
| Overlap | 두 화자 발화 시간 겹침 | `< 겹친 말 > [<]` / `< 겹친 말 > [>]` |
| 빠른 발화 | 음절 속도 > 중앙값 + 2×MAD | `∆ ... ∆` |
| 느린 발화 | 음절 속도 < 중앙값 − 2×MAD | `∇ ... ∇` |
| 발화 병합 | 같은 화자, 침묵 < 0.1초면 한 발화 | — |

> **다른 맥에 인계·여러 대로 나눠 돌리기**:
> [인계_다른맥_설치.md](인계_다른맥_설치.md)
> — `.venv`는 복사하면 깨진다, HF 토큰은 각자 발급,
> 켜면 안 되는 옵션(`--fast`·`--parallel-diarize`) 등.

> **개발 배경·설계 근거·실측 발견**:
> [개발기록_및_실측발견.md](개발기록_및_실측발견.md)
> — 왜 이렇게 고쳤는지와 측정으로 확인된 것들을 모아둔 문서.
> VAD 임계값을 *낮추면* 오히려 발화가 누락되는 반직관적 구간,
> 기존 녹음이 dual-mono라 겹침말 복원이 원리적으로 불가능한 점,
> 한국어 교착어 형태론이 cut-off 검출을 무력화하는 문제 등.

> **장시간 배치를 돌리기 전에 반드시 읽을 것**:
> [배치_실행_주의사항.md](배치_실행_주의사항.md)
> — 절전 차단(`caffeinate -dimsu`)을 빠뜨리면 맥이 밤새 자면서
> 처리 속도가 **10배 느려진다**(실측). 세션 분리 실행·자동 재시작·
> dual-pass 사용 기준도 함께 정리돼 있다.

## 설치 (Mac, 1회만)

```bash
cd gailbot-lite
bash install_mac.sh
```

이게 전부입니다. 스크립트가 가상환경(.venv)을 만들고 faster-whisper를 설치합니다.
Homebrew, conda, ffmpeg 모두 **필요 없습니다**. Python 3.9 이상만 있으면 됩니다
(macOS 최신 버전엔 기본 포함).

## 사용법

터미널에서 매번 먼저 가상환경을 켜세요:

```bash
cd gailbot-lite
source .venv/bin/activate
```

### 1) 화자별로 녹음 파일이 따로 있을 때 (권장 — CA 연구 표준)

```bash
python -m gblite transcribe speakerA.wav speakerB.wav -o results/
```

파일 1개 = 화자 1명으로 처리되고, 파일명이 화자 이름이 됩니다.
두 파일의 타임라인이 같다면(같은 녹음의 채널 분리) overlap까지 정확히 잡힙니다.

### 2) 파일이 하나뿐일 때 (화자 1명으로 처리)

```bash
python -m gblite transcribe interview.mp3 -o results/ --language ko
```

wav, mp3, m4a, mp4, flac 등 대부분의 형식을 그대로 넣을 수 있습니다.

### 2-1) 화자가 섞인 단일 녹음을 자동으로 분리 (diarization)

한 파일 안에 여러 사람이 섞여 있으면 `--diarize`로 자동 분리합니다.
먼저 추가 설치가 한 번 필요합니다 (아래 "화자 분리 설치" 참고):

```bash
# 화자 수를 알면 지정하는 게 정확합니다
python -m gblite transcribe 인터뷰.mp3 --diarize --num-speakers 2 -o results/

# 몇 명인지 모르면 자동 감지 (범위만 줄 수도 있음)
python -m gblite transcribe 회의.m4a --diarize --min-speakers 2 --max-speakers 5 -o results/
```

분리된 화자는 `SP_A`, `SP_B` … 로 라벨링되고, 화자가 바뀌는 지점은 gap,
같은 화자 안의 침묵은 pause로 자동 표시됩니다.

### 2-2) 폴더 통째로 일괄 처리 (여러 녹음을 한 번에)

mp3/m4a 파일이 여러 개 든 폴더를 한 번에 돌립니다. **파일 하나하나가
별개의 녹음으로 독립 처리**되고, 결과도 파일별 폴더로 나뉩니다.

```bash
# 예: audio 폴더 안 모든 오디오를 처리
python -m gblite batch audio/ -o results/

# 각 파일마다 화자 자동 분리까지
python -m gblite batch audio/ --diarize --num-speakers 2 -o results/
```

zip으로 받았다면 먼저 압축을 풀고 그 폴더를 넣으면 됩니다:

```bash
unzip mp3.zip -d audio/        # audio/ 폴더에 풀기
python -m gblite batch audio/ -o results/
```

결과는 이렇게 파일별로 정리돼요:

```
results/
  ├── 녹음01/  (conversation.txt, .cha, *_turns.csv, utt.toml ...)
  ├── 녹음02/
  └── 녹음03/
```

중간에 한 파일이 깨져도 멈추지 않고 건너뛴 뒤, 마지막에 성공/실패 개수를
알려줍니다. (지원 형식: wav, mp3, m4a, mp4, flac, aac, ogg, opus 등)

### 3) 이미 전사된 결과에 CA 주석만 붙일 때

기존 GailBot이 만든 `utt.toml`(단어 단위)이나 CSV가 있다면 STT 없이 바로:

```bash
python -m gblite annotate utt.toml -o results/
```

### 주요 옵션

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `-m, --model` | Whisper 모델: tiny/base/small/medium/large-v3/large-v3-turbo | `small` |
| `-l, --language` | 언어 코드 (`en`, `ko` 등). 생략 시 자동 감지 | 자동 |
| `-o, --output` | 결과 폴더 | `gblite_results` |
| `--name` | 출력 파일 이름 접두어 | `conversation` |
| `--diarize` | 단일 혼합 녹음 화자 자동 분리 | 꺼짐 |
| `--num-speakers` | 화자 수 지정 (정확도↑) | 자동 |
| `--min-speakers` / `--max-speakers` | 화자 수 범위 | — |
| `--hf-token` | HuggingFace 토큰 (또는 `HF_TOKEN` 환경변수) | — |

정확도가 더 필요하면 `-m medium` 또는 `-m large-v3-turbo`를 쓰세요
(Apple Silicon 16GB 메모리면 충분히 돌아갑니다).

## 화자 분리 설치 (선택 — `--diarize` 쓸 때만)

기본 설치는 가볍게 유지하려고 화자 분리 패키지를 뺐습니다
(pyannote + torch는 용량이 크고, 원본 GailBot 설치가 깨지던 주범이기도 해서요).
필요할 때만 추가로 설치하세요:

```bash
cd gailbot-lite
bash install_mac.sh --diarize
```

그리고 **1회만** HuggingFace 설정이 필요합니다 (무료):

1. https://huggingface.co/settings/tokens 에서 토큰(Read 권한) 발급
2. 아래 두 모델 페이지에서 **Agree** 클릭 (약관 동의):
   - https://huggingface.co/pyannote/speaker-diarization-3.1
   - https://huggingface.co/pyannote/segmentation-3.0
3. 실행할 때 토큰 전달:

```bash
export HF_TOKEN=hf_xxxxxxxx        # 터미널 세션마다 한 번
python -m gblite transcribe 회의.mp3 --diarize --num-speakers 3 -o results/
# 또는 매번:  --hf-token hf_xxxxxxxx
```

토큰이나 약관 동의가 빠지면 실행 시 무엇을 해야 하는지 안내 메시지가 나옵니다.
파일이 화자별로 이미 나뉘어 있다면 `--diarize` 없이 여러 파일을 넣는
방식(위 1번)이 더 정확하니 그쪽을 권합니다.

## 결과물

| 파일 | 내용 |
|---|---|
| `*.txt` | CA 주석이 들어간 사람이 읽는 전사문 (화자별 턴 + 타임스탬프) |
| `*.cha` | CHAT/CLAN 형식 (TalkBank 호환, 밀리초 정렬 불릿 포함) |
| `*_turns.csv` | 턴 단위 표 (화자, 텍스트+주석, 시작/끝 시간) |
| `*_words.csv` | 단어 단위 표 |
| `*_markers.csv` | 검출된 모든 CA 마커 목록 |
| `utt.toml` | 원본 GailBot과 같은 형식의 단어 단위 출력 (재처리용) |
| `*_stats.json` | 음절 속도 통계 (중앙값, MAD, 빠른/느린 턴 수) |

## 자주 겪는 문제

- **첫 실행이 느려요** → 처음 한 번 Whisper 모델을 다운로드합니다(small ≈ 460MB).
  이후에는 캐시에서 바로 로드됩니다.
- **`pip: command not found`** → `source .venv/bin/activate`를 먼저 실행했는지 확인.
- **화자 자동 분리(diarization)** → `--diarize` 옵션으로 지원합니다
  (위 "화자 분리 설치" 참고). 기본 설치에는 빠져 있고, 필요할 때만
  `bash install_mac.sh --diarize`로 추가합니다.

## 원본과의 차이

- IBM Watson / Google STT 제거 → 로컬 Whisper만 사용 (API 키, 과금 없음)
- PyQt GUI 없음 → 터미널 CLI
- diarization은 **선택 설치**로 분리 (기본은 파일별 화자 지정, `--diarize`로 자동 분리)
- BST/플러그인 프레임워크 제거 → 같은 로직을 단순한 함수로 재작성
- CA 기준값·마커 표기는 원본 HiLabSuite `configData.toml`과 동일

라이선스: MIT (원본 GailBot의 분석 로직을 참고하여 재구현; 원본 저작권은
Human Interaction Lab, Tufts University).
연구 출판 시 원본 GailBot 논문 인용을 권장합니다:
Umair, Mertens, Albert & de Ruiter (2022), *GailBot: An automatic transcription
system for Conversation Analysis*, Dialogue & Discourse 13(1).

## 공개 범위

이 저장소는 **코드와 기준값 문서만** 담는다. 연구 참여자의 음원·전사·정답지·세션 식별자는 IRB 동의 범위에 따라 비공개이며,
코드 주석의 참여자 표기는 `참여자A/B…`, 세션은 `S-nn`으로 익명화했다. 배치 실행 스크립트·NAS 패키징·검증용 원장은 연구실 내부에만 있다.
