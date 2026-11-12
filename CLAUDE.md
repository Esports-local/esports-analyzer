# esports-analyzer — 분석 파이프라인 (FastAPI)

> 프로젝트 전체 구성, 결정 사항, 저장소 구조는 **`docs/PLAN.md`가 기준**이다 (api·ui 레포도 이 파일을 본다). 작업 전에 반드시 읽는다.
> 주제: LoL 한국어 중계 → 영어 번역에서 화면 글자(OCR)로 고유명사 오역을 줄이는 실험.
> 드라마 스토리 예측 버전(faces·similarity·scenes·vlm·episode 단계)은 `drama` 브랜치에 있다.
> api ↔ analyzer 연동 스펙은 `api/<엔드포인트>/schemas.py`와 docs/PLAN.md 6장이 기준이다 (`/videos`, `/jobs`, `/estimate`). 바꾸면 esports-api 의 `api/games/analyzer.py`도 같이 고친다.

## 이 레포의 역할
중계 영상 분석 파이프라인. HTTP 서버는 작업 접수/조회/취소만 하는 얇은 층이고, 실제 처리는 `pipeline/`에서 한다.
분석은 수십 분 걸리는 작업이므로 FastAPI BackgroundTasks가 아니라 큐 + 워커로 실행할 예정.

## 구조
- `pipeline/stages.py`: 단계 목록과 `run_stage`, `run_pipeline` (코드에서 실행: `from pipeline import run_pipeline`). 각 단계는 `outputs/<영상명>/work/<stage>.json` 출력 → 중간부터 재실행 가능
  - 단계: preprocess → segments → hls → roi_ocr → asr → translate → evaluate(예정)
  - hls 는 `outputs/<영상명>/hls/` (master.m3u8, index.m3u8, seg_00001.ts …)
- `pipeline/roi_ocr.py`: 중계 화면 고정 영역 OCR (RapidOCR 글자 인식만). 설정은 `configs/<종목>/<레이아웃>.json`
  (`ocr_roi` [x1,y1,x2,y2] 1920x1080 기준, `allowed_text`, `validation`, `lexicon`, `static`), 사전은 `configs/<종목>/lexicon/<이름>.json`(선택, 지금은 팀 로고 글꼴 보정용 `teams.json`만).
  결과 `work/roi_ocr.json`: 영역(tag+slot)마다 같은 값이 이어진 구간. 화면이 그대로면 OCR 을 다시 돌리지 않는다
- `pipeline/asr.py`: faster-whisper (Silero VAD 내장, 대사 단위 언어 감지). 해설 대사 → `work/asr.json`
- `pipeline/translate.py`: 조건 A(대사)·B(+화면 OCR)·C(+프레임) 영어 번역, Gemini REST (`GEMINI_API_KEY`). 기본 A·B, `dry_run`(입력 토큰만 셈, 무료), 기존 번역 재사용. 화면 정보는 `pipeline/screen.py`
- `pipeline/__main__.py`: CLI. `--set 단계.키=값`으로 설정 덮어쓰기 (예: `--set translate.dry_run=true`)
- `app/main.py`: FastAPI 앱. 엔드포인트 router 를 묶기만 한다 (`uvicorn app.main:app`)
- `api/`: HTTP 엔드포인트. 엔드포인트마다 폴더 하나 (`views.py` router + `schemas.py` 요청·응답 모양 = 연동 스펙). 큐는 아직 없음
  - 공통: `views.py`(`GET /health`), `paths.py`(`VIDEOS_ROOT`·`OUTPUTS_ROOT` 절대 경로, 영상 경로 확인. 결과 폴더 이름은 요청한 파일 이름 기준이라 바로가기여도 api 의 work_name 과 같다)
  - `videos/`: `GET /videos` (VIDEOS_ROOT 영상, 1920x1080·소리 있는 것만 selectable)
  - `jobs/`: `POST /jobs`(파이프라인을 프로세스로 실행, 한 번에 하나, 로그 `work/job.log`), `GET /jobs/{id}`, `POST /jobs/{id}/cancel`
    - 같은 job_id 를 다른 내용으로 보내면 409. 취소는 프로세스 그룹째(ffmpeg 등 하위 프로세스까지). 끝난 작업 기록은 최근 100개만 남김
  - `estimate/`: `POST /estimate` (번역 입력 토큰)

```
docker build -t esports-analyzer .
docker run --rm -p 8001:8001 -v "$PWD":/app esports-analyzer uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
docker run --rm -v "$PWD":/app -v ~/vod-data:/data esports-analyzer python -m pipeline /data/game.mp4 --only preprocess,roi_ocr,asr
```
현재 Dockerfile은 CPU용. GPU가 필요해지면 CUDA 베이스 이미지로 별도 작성.

## 모델
TransNetV2(화면 전환 검출, CPU) · RapidOCR(글자 인식, CPU) · faster-whisper(ASR, Silero VAD 내장) · pyannote Community-1(화자 구분, 선택, HF_TOKEN)
· 번역 Gemini (`gemini-3.8-flash`, 조건 A/B/C 같은 모델·프롬프트, docs/PLAN.md 2장).

## 규칙 요약 (상세는 docs/PLAN.md)
- 시간은 초 단위 float, 단계 결과 JSON에는 `schema_version`
- `work/`: 단계별 중간 결과 (외부 공개 안 함) · `hls/`: 재생용
- 화면 좌표는 1920x1080 기준 픽셀. 다른 해상도 영상은 그 크기로 맞춰 읽는다
- 패키지는 `requirements.txt`에 `==` 고정 후 이미지 재빌드 (로컬 가상환경 안 씀)
- 중계 영상은 연구용으로 로컬에서만 쓰고 재배포하지 않는다 (`*.mp4`, `data/`, `outputs/`는 git 제외)

## 커밋 규칙
- 메시지는 `종류/내용` 한 줄로 간단하게 쓴다
  - `feat/` 기능 구현 · `refactor/` 구조 변경 · `fix/` 버그 수정 · `docs/` 문서
  - 예: `feat/TransNetV2 segments 단계 구현`
- 작성자는 사용자 본인만. Claude 를 `Co-Authored-By` 등 공동 작성자로 넣지 않는다
