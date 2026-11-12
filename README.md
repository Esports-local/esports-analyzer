# esports-analyzer

LoL 한국어 중계 → 영어 번역에서 **화면 글자(OCR)로 고유명사 오역을 줄이는 실험**의 분석 파이프라인.
중계 녹화본에서 해설 대사(ASR)와 화면 정보(팀·선수 이름·킬·KDA·골드·시간 OCR)를 뽑는다.

- 계획·실험 조건·지금까지 결과: [`docs/PLAN.md`](docs/PLAN.md)
- 남은 작업: [`CHECKLIST.md`](CHECKLIST.md)
- 함께 쓰는 레포: [esports-api](https://github.com/djkim0704/esports-api) (결과 저장·조회) · [esports-ui](https://github.com/djkim0704/esports-ui) (화면)

## 단계
| 단계 | 내용 | 결과 |
|---|---|---|
| preprocess | 영상 정보, 16kHz wav | `work/preprocess.json`, `work/audio.wav` |
| segments | 화면 전환 검출 (TransNetV2) | `work/segments.json` |
| hls | 재생용 ts | `hls/` |
| roi_ocr | 고정 영역 OCR (RapidOCR 인식만), 같은 값이 이어진 구간 | `work/roi_ocr.json` |
| asr | faster-whisper (Silero VAD), 대사 | `work/asr.json` |
| translate · evaluate | 조건별 번역, 고유명사 평가 | 예정 |

ROI 설정은 `configs/lol/worlds_2026.json` (좌표는 1920x1080 기준).

## 실행 (Docker, CPU)
```
docker build -t esports-analyzer .
docker run --rm -v "$PWD":/app -v /path/to/videos:/data esports-analyzer \
  python -m pipeline /data/game.mp4 --only preprocess,roi_ocr,asr
```
결과는 `outputs/<영상 파일명>/work/`에 생긴다. 번역 테스트에는 `.env`(`cp .env.example .env`)의 `GEMINI_API_KEY`가 필요하다.

화면 녹화본이 1920x1080이 아니면 중계 화면 부분만 잘라 1080p로 변환해서 쓴다:
```
ffmpeg -i rec.mov -vf "crop=W:H:X:Y,scale=1920:1080:flags=lanczos,fps=30" -c:v libx264 -crf 18 -c:a copy game.mp4
```

## 영상 데이터
중계 영상은 연구용으로 로컬에서만 쓰고 이 레포에 올리지 않는다 (`*.mp4`, `data/`, `outputs/`는 git 제외).
