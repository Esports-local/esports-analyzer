# test_main — 분석 파이프라인 (FastAPI)

> 프로젝트 전체 구성, 결정 사항, 저장소 구조는 **`../PLAN.md`가 기준**이다. 작업 전에 반드시 읽는다.
> api ↔ main 연동 스펙은 아직 미정이다. 현재 `app/`의 엔드포인트와 필드 이름은 임시이며, 확정 전에 바꾸지 말고 파이프라인 구현에 집중한다.

## 이 레포의 역할
VOD 분석 파이프라인. HTTP 서버는 작업 접수/조회/취소만 하는 얇은 층이고, 실제 처리는 `pipeline/`에서 한다.
분석은 수십 분 걸리는 작업이므로 FastAPI BackgroundTasks가 아니라 큐 + 워커로 실행할 예정.

## 구조
- `pipeline/stages.py`: 단계 목록과 `run_stage`, `run_pipeline` (코드에서 실행: `from pipeline import run_pipeline`). 각 단계는 `outputs/<영상명>/work/<stage>.json` 출력 → 중간부터 재실행 가능
  - hls 는 `outputs/<영상명>/hls/` (master.m3u8, index.m3u8, seg_00001.ts …)
- `pipeline/__main__.py`: CLI. `python -m pipeline /data/ep01.mp4 --until hls`
- `app/`: FastAPI (임시)

```
docker build -t vod-main .
docker run --rm -p 8001:8001 -v "$PWD":/app vod-main uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
docker run --rm -v "$PWD":/app -v ~/vod-data:/data vod-main python -m pipeline /data/ep01.mp4 --until segments
```
현재 Dockerfile은 CPU용. GPU 서버용은 CUDA 베이스 이미지로 별도 작성 예정.

## 모델 (설계안)
TransNetV2(전환 검출: 컷·디졸브·와이프·페이드, CPU) · InsightFace buffalo_l(얼굴 검출·인식, CPU, 비상업 연구용) · open_clip 이미지 임베딩(화면 유사도, 기본 ViT-B-32, MobileCLIP 은 이름·라이선스 확인 후 교체) · faster-whisper(ASR, Silero VAD 내장) · pyannote Community-1(화자 구분, 선택, HF_TOKEN)
· Qwen3-VL 2B/4B 4bit(VLM, GPU) · 예측 LLM은 미정.

## 개발 제약
로컬 맥에는 GPU 없음. VLM을 뺀 단계는 CPU로 로컬 개발 가능.
VLM은 백엔드 인터페이스로 분리: `mock`(로컬) / `mlx`(Apple Silicon, 가능 여부 확인 필요) / `transformers`·`vLLM`(GPU 서버).

## 작업 순서
1. preprocess (ffprobe + 16kHz wav) → segments (전환 검출) → hls (전환 기준 ts ≤4초) → faces (얼굴 군집 F1/F2…) (짧은 클립으로 검증)
2. asr (word_timestamps 사용, 컷을 넘는 대사는 자르지 않음, 화자 라벨 S1/S2…)
3. similarity (segment 대표 프레임을 직접 뽑아 비교) → scenes
   - 라벨링 결과 확인은 별도 레포의 웹에서 한다 (이 레포에는 검수 HTML 을 두지 않음)
4. vlm (mock부터) → episode (최종 `analysis/episode.json`)
   - 예측(1차 LLM: header → body 선별 → 2차 LLM)은 api 쪽에서 한다
5. 큐 + 워커, api 연동 (스펙 확정 후)

## 규칙 요약 (상세는 PLAN.md)
- 시간은 초 단위 float, 단계 결과 JSON에는 `schema_version`
- 결과 파일은 MinIO 구조(`hls/`, `analysis/`)에 맞춰 저장 (썸네일은 쓰지 않으므로 `frames/` 없음)
  - `work/`: 단계별 중간 결과·키프레임 (외부 공개 안 함) · `hls/`: 영상 구조(m3u8) · `analysis/`: 영상 내용(JSON)
- 패키지는 `requirements.txt`에 `==` 고정 후 이미지 재빌드 (로컬 가상환경 안 씀)

## 커밋 규칙
- 메시지는 `종류/내용` 한 줄로 간단하게 쓴다
  - `feat/` 기능 구현 · `refactor/` 구조 변경 · `fix/` 버그 수정 · `docs/` 문서
  - 예: `feat/TransNetV2 segments 단계 구현`
- 작성자는 사용자 본인만. Claude 를 `Co-Authored-By` 등 공동 작성자로 넣지 않는다
