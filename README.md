# esports-analyzer

LoL 중계 영상에서 **경기 정보 · 화면 OCR · 해설 받아쓰기로 시간별 경기 데이터를 만들고 숏폼을 만드는** 분석 파이프라인.
경기 정보(header)는 op.gg + Data Dragon 으로 자동으로 만들고, 영상에서는 장면 전환·화면 종류, 화면 숫자(킬·KDA·골드·시간·오브젝트), 해설 대사를 뽑아 `analyzer.json` 하나로 묶는다. LLM 은 쓰지 않는다.

- 계획·실험 조건·지금까지 결과: [`docs/PLAN.md`](docs/PLAN.md)
- 남은 작업: [`CHECKLIST.md`](CHECKLIST.md) (번역 실험 때 쓴 것이라 낡음, 다시 쓸 예정)
- 함께 쓰는 레포: [esports-api](https://github.com/Esports-local/esports-api) (결과 저장·조회) · [esports-ui](https://github.com/Esports-local/esports-ui) (화면)

## 단계
| 단계 | 내용 | 결과 |
|---|---|---|
| header | op.gg 경기·세트 + Data Dragon → 팀·선수·챔피언·밴·결과 (영상 안 읽음, `match`·`set` 필요). 같은 요청으로 op.gg 세트 기록(검증 정답지) | `work/header.json`, `work/opgg_timeline.json` |
| preprocess | 영상 정보, 16kHz wav | `work/preprocess.json`, `work/audio.wav` |
| segments | 장면 전환 (cut · dissolve · wipe, TransNetV2) + 구간마다 화면 종류 (live · replay · postgame · countdown · draft · break, LCK) + 경기 구간 | `work/segments.json` |
| roi_ocr | 고정 영역 OCR (RapidOCR 인식만), 같은 값이 이어진 구간. 경기 밖·리플레이 구간은 건너뜀 | `work/roi_ocr.json` |
| asr | faster-whisper small (CPU int8, Silero VAD), 대사 + header 단어로 고유명사 교정. 밴픽·카운트다운·쉬는 시간은 무음 처리 | `work/asr.json` |
| timeline | 화면 숫자(팀 킬·타워·오브젝트)가 오른 시점 → 경기 이벤트. 킬은 KDA 변화로 killer·victims·assists, LCK 는 배너·용 아이콘으로 오브젝트 종류·팀 | `work/timeline.json` |
| context | 바디: 시계(영상 초 ↔ 경기 시각), 구간, 이벤트 태그(first_blood·multikill·teamfight), 초 단위 신호(소리 크기·골드 차이·컷 빈도) | `work/context.json` |
| evaluate | 고유명사 평가 | 예정 |

단계가 끝날 때마다 `outputs/<영상>/analyzer.json`(시간순 트리: phase → 장면 → 이벤트·대사·op.gg 정답지 짝·신호)으로 묶는다. 예: [`docs/examples/lck_final_g1_900-1500s/analyzer.json`](docs/examples/lck_final_g1_900-1500s/analyzer.json)

10분 구간(LCK 결승 900~1500초, CPU 10개) 처리 시간: segments 327초 · roi_ocr 152초 · asr 187초. asr 은 CPU 에서 float32 로 돌던 것을 int8 로 바꿔 351초 → 187초, 메모리 2.7GB → 1.4GB.

숏폼(`pipeline/shorts/render.py`)은 단계가 아니라 요청마다 만든다: 고른 이벤트 구간 → 세로 720x1280 mp4 (중계 화면 가운데 9:16 잘라내기, 자막 없음) → `shorts/<key>.mp4`. 17초 숏폼 렌더 약 3초 (CPU).

화면 좌표(1920x1080 기준)는 레이아웃 파일 하나에 둔다: `configs/lol/lck_2026.json`(치지직 LCK, 화면 종류 규칙 포함) · `worlds_2026.json`. header 의 리그로 `layouts.json`에서 고르고, 새 대회는 `_template.json`을 복사해 채운다. 좌표 확인 그림: `python -m pipeline.screen.layout <영상> <초> --out p.png`.

## HTTP (:8001)
esports-api 가 부른다. 요청·응답 모양은 엔드포인트 폴더의 `schemas.py`가 연동 스펙이다.

| 메서드 | 경로 | 설명 | 폴더 |
|---|---|---|---|
| GET | `/health` | 상태, 단계 목록 | `api/views.py` |
| GET | `/videos` | 원본 영상 목록 (1920x1080·소리 있는 것만 selectable) | `api/videos/` |
| GET | `/videos/{name}/file` | 원본 영상 파일 (Range 요청, 브라우저 재생용) | `api/videos/` |
| POST | `/jobs` | 분석 작업 시작 (한 번에 하나, 같은 job_id 를 다른 내용으로 보내면 409) | `api/jobs/` |
| GET | `/jobs/{id}` | 작업 상태 (현재 단계, 경과 시간, 실패 시 에러) | `api/jobs/` |
| POST | `/jobs/{id}/cancel` | 작업 중지 (ffmpeg 등 하위 프로세스까지) | `api/jobs/` |
| POST | `/shorts` | 숏폼 렌더 접수. 다른 렌더가 돌고 있으면 줄을 세운다. 같은 key·내용은 기존 상태, 실패한 것은 다시 렌더 | `api/shorts/` |
| GET | `/shorts/{key}` | 숏폼 상태 (대기 순서, 렌더 시간, 결과 경로) | `api/shorts/` |

`app/main.py`는 FastAPI 앱을 만들고 router 를 묶기만 한다. 작업 기록은 메모리에만 있어서 서버를 다시 띄우면 사라진다 (결과 파일은 남는다).

## 실행
보통은 `esports-api`의 `docker compose up`으로 ui(:8000)·api(:8002)와 같이 띄우고, ui의 "영상 추가" 화면에서 영상을 골라 분석한다.

### 직접 실행 (Docker, CPU)
```
docker build -t esports-analyzer .
docker run --rm -v "$PWD":/app -v /path/to/videos:/data esports-analyzer \
  python -m pipeline /data/game.mp4 --start-sec 900 --max-sec 600 \
  --only header,preprocess,segments,roi_ocr,asr,timeline,context --set header.match=32655 --set header.set=1
```
결과는 `outputs/<영상 파일명>_900-1500s/`에 생긴다 (구간을 안 주면 `outputs/<영상 파일명>/`). 결과 시각은 원본 영상 기준이다.

화면 녹화본이 1920x1080이 아니면 중계 화면 부분만 잘라 1080p로 변환해서 쓴다:
```
ffmpeg -i rec.mov -vf "crop=W:H:X:Y,scale=1920:1080:flags=lanczos,fps=30" -c:v libx264 -crf 18 -c:a copy game.mp4
```

## 영상 데이터
중계 영상은 연구용으로 로컬에서만 쓰고 이 레포에 올리지 않는다 (`*.mp4`, `data/`, `outputs/`는 git 제외).
