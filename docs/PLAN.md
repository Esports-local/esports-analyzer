# e스포츠 중계 번역 + 화면 OCR — 프로젝트 계획

세 레포(`esports-analyzer` · `esports-api` · `esports-ui`)의 기준 문서. 이 파일은 `esports-analyzer/docs/PLAN.md`에 둔다.
종목은 **리그 오브 레전드(LoL)**, 번역 방향은 **한국어 중계 → 영어**.
이전 주제(드라마 스토리 예측)는 각 레포의 `drama` 브랜치에 있다.

## 1. 목표

e스포츠 중계를 **음성만으로 번역하면 선수명·팀명·챔피언명·스킬명이 자주 틀린다.**
화면에 이미 나와 있는 글자를 OCR로 읽어 번역 프롬프트에 넣으면 이 오역이 얼마나 줄어드는지 측정한다.

1. 음성만으로 번역했을 때 생기는 고유명사 오역 사례를 모으고, **화면 정보로 바로잡을 수 있는 사례**를 골라 유형별 케이스셋을 만든다
2. 화면 글자(OCR)를 번역 프롬프트에 넣었을 때 오역이 얼마나 줄어드는지 측정한다
3. 영상을 모델에 직접 넣는 방식(VLM)과 비교해서, **어떤 화면 구성·상황에서 효과가 큰지** 분석한다

## 2. 실험 조건

같은 대사(ASR 결과)를 같은 번역 모델로 조건만 바꿔 번역한다. ASR 결과는 모든 조건에 같은 것을 쓴다.

| 조건 | 번역 입력 | 비고 |
|---|---|---|
| A | ASR 대사 (앞뒤 2줄 포함) | 기준선 |
| B | A + 그 시점 화면 OCR (팀, 선수 10명, 킬, KDA, 게임 시간) | 이 연구의 제안 방식 |
| C | A + 그 시점 영상 프레임 | VLM |
| B+ (선택) | A + 경기 정보(화면 밖 정답 자료) | 정보가 완벽할 때의 상한 |
| B' (선택) | LLM 없이 규칙으로 이름만 치환 | 비교 기준 |

- 번역 모델: Gemini (텍스트·이미지·영상을 다 받아서 A·B·C를 한 모델로 맞출 수 있음). 모델·프롬프트·온도는 조건 간 고정
- 오역 원인을 **ASR 단계(이름을 잘못 들음)**와 **번역 단계(들었지만 잘못 옮김)**로 나눠 기록한다
- 주의: LLM이 유명 선수(T1 등)를 이미 알고 있어 조건 A에서도 이름을 고친다. 잘 알려지지 않은 대상(덜 유명한 선수·챔피언·스킬)을 따로 집계한다

## 3. 데이터

- 영상: 중계 화면 녹화본 (OBS, 1920x1080 권장). **연구용으로 로컬에서만 쓰고 재배포하지 않는다** (`*.mp4`, `*.mov`, `data/`, `outputs/`는 git 제외)
  - 지금: Worlds 2026 4강 BLG vs T1 2세트 — 한국어 해설 17분, 영어 해설 10분
  - 영어 해설은 고유명사 표기 참고용 (문장 단위 대응 번역본은 아님)
- 경기 정보: analyzer `header` 단계가 만든다 (api 는 op.gg 를 부르지 않고 결과만 저장·제공)
  - op.gg 경기 페이지(`esports.op.gg/matches/<번호>`)의 페이지 안 JSON: 경기·팀·스코어·세트 영상 링크
  - 세트별 진영·선수·챔피언·밴·승자: op.gg 세트 정보(`/matches/graphql` GetGameByMatch, 경기 페이지 HTML에는 없음). 챔피언 번호 → 이름·스킬은 Riot Data Dragon
  - 피어리스(LCK·LPL·MSI·Worlds, 2025년 이후): 앞 세트 픽은 이 세트에서 못 고르는 챔피언으로 헤더에 넣는다
- 챔피언 한국어 ↔ 영어: Riot Data Dragon (`ko_KR` / `en_US`)

## 4. 파이프라인 (esports-analyzer)

```
영상 ─▶ preprocess ─▶ asr ──────────────────────┐
          │                                      ▼
          └──▶ roi_ocr (영역 OCR → 구간) ──▶ translate (A/B/C) ─▶ evaluate
                                                 ▲
                              영상 프레임 (C) ───┘
```

| 단계 | 내용 | 상태 |
|---|---|---|
| header | 경기 기본 정보 (op.gg + Data Dragon, 영상 안 읽음, `match`·`set` 설정) → `work/header.json` | 완료 |
| preprocess | 영상 정보, VFR 점검, 16kHz wav | 완료 |
| segments | 장면 전환 (cut · dissolve · wipe, TransNetV2) + 화면 종류 (live · replay · postgame · break) + 경기 구간 | LCK 규칙 완료 |
| roi_ocr | 고정 영역 OCR (RapidOCR 인식만) → 같은 값이 이어진 구간 | 완료 (실제 영상 확인) |
| asr | faster-whisper (Silero VAD) → 대사 | 완료 (실제 영상 확인) |
| translate | 조건별 번역 (Gemini) | 지움 (git 기록) |
| evaluate | 고유명사 오역 판정, 조건별 집계 | 예정 |

### 레이아웃 설정 (`configs/lol/<대회>.json`, 리그로 자동 선택 `configs/lol/layouts.json`)
`rois`(roi_ocr 영역)와 `screens`(segments 화면 종류 규칙, LCK 만)를 한 파일에 둔다. 영역 하나에 `tag`, `slot`, `kind`, `ocr_roi` [x1, y1, x2, y2] (1920x1080 기준), `ocr_scale`, `char_map`, `allowed_text`,
`validation` (`regex`, `min_confidence`), `static`(경기 중 안 바뀌는 값: 5초 이어지면 확정).
템플릿 게이트는 쓰지 않는다. 리플레이·캠 화면은 검증(허용 문자·형식·신뢰도)에서 걸러진다.

### 결과 (`outputs/<영상명>/work/`)
- `roi_ocr.json`: `runs[]` = {tag, slot, kind, text, raw, start_sec, end_sec, conf, samples}
- `asr.json`: `lines[]` = {id, start_sec, end_sec, speaker, language, text}
- `run_log.json`: 단계별 시간·메모리

## 5. 평가와 케이스셋

- 정답: 대사마다 등장하는 고유명사의 영문 정식 표기 (`name`). 경기 정보 + 화면 OCR로 만들고 사람이 확인
- 지표: 고유명사 정확도, A 대비 B·C의 오역 감소율, 오역 원인(ASR / 번역) 비율
- 분석 축: 종류(선수·팀·챔피언·스킬) · 화면 상태(경기 / 리플레이 / 캠) · 화면 근거 유무
- 화면 근거로 쓸 수 있는 사건: 선수별 KDA 변화(누가 언제 죽었나) + 포지션 → 챔피언, 죽은 챔피언 초상화(흑백·부활 시간)

| 유형 | 예 | 화면 근거 |
|---|---|---|
| 선수명 | 케리아 → "캐리아", Xun → "Shun" | 선수 이름 칸 (글자) |
| 팀명 | 티원 → "티언" | 팀 칸 (글자) |
| 챔피언명 | 키아나 → "오브로키아나" | 아이콘뿐 (C나 경기 정보 필요) |
| 스킬·아이템명 | 공식 명칭 대신 직역 | 아이콘뿐 |

## 6. 레포와 연동

| 레포 | 역할 |
|---|---|
| esports-analyzer | 분석 파이프라인 (FastAPI는 임시) |
| esports-api | 경기 정보(op.gg)·세트 영상·분석 결과 저장, ui에 제공 (Django + DRF, 로컬 sqlite) |
| esports-ui | 세트 영상 목록, 대사·번역 비교와 대사 검수, 화면 OCR 구간, 숏폼(이벤트 고르기·재생), 원본 JSON 화면 (Django 템플릿) |
| dev-workflow | 세 레포에서 쓰는 Claude Code 플러그인 `esports-dev`(스킬 9개)와 스킬 선택 평가·설명 자동 개선·분류. 연구 주제와는 별개 |

- ui → api 스펙: [스펙 시트 spec_sheet_lol](https://docs.google.com/spreadsheets/d/1ktzJNw1eFwdPfRUiwNkzmhVgz8ZyxqiYMk721m9E6C8)
  - 1 `GET /games` · 2 `GET /games/{id}` · 3 `/screen` · 4 `/lines` · 5 `/analysis` · 8 `POST /games`(영상 등록 + 분석) · 9 `/analyze` · 10 `/cancel` · 12 `GET /sources` · 13 `POST /lines/{line_id}/review`(대사 검수: 고친 문장·O/X, 고친 문장으로 번역) · 14 `GET /games/{id}/events`(경기 이벤트 + 숏폼 기본 구간) · 15 `POST /games/{id}/shorts`(숏폼 만들기, 같은 요청은 기존 숏폼) · 16 `GET /games/{id}/shorts` · 17 `GET /shorts/{id}`(상태·재생 주소) (구현) · 6·7 `/cases` (평가 단계 후)
  - `/skills` (스펙 시트 번호 없음): 스킬 사용 기록·선택 평가. ui `/skills/` 화면이 읽고, dev-workflow 의 훅·`evals/run.py`·`evals/worker.py`가 보낸다
- api → analyzer (`api/*/schemas.py`): `GET /videos`(원본 영상, 1920x1080·소리 있는 것만 selectable) · `POST /jobs`(파이프라인 실행, 한 번에 하나) · `GET /jobs/{id}` · `POST /jobs/{id}/cancel` · `POST /shorts`(숏폼 렌더 접수, 다른 렌더가 돌고 있으면 줄 세움) · `GET /shorts/{key}`
  - api 는 세트 영상을 조회할 때 analyzer 작업 상태를 묻고, 끝났으면 결과 폴더(`outputs/<이름>/work/`)를 읽어 저장한다
  - 큐는 아직 없다. analyzer 컨테이너를 다시 띄우면 진행 중이던 작업 기록이 사라진다
  - 숏폼: api 가 key(입력 해시)를 정해 analyzer 에 렌더를 맡기고, 조회할 때 상태를 묻는다. 끝나면 analyzer 결과 폴더의 mp4(`outputs/<이름>/shorts/<key>.mp4`)를 MinIO `shorts/<key>.mp4`(공개 읽기)로 올린다. analyzer 를 다시 띄워 기록이 없으면(404) 같은 내용으로 다시 맡긴다
- 실행: `esports-api`에서 `docker compose up` (ui :8000 + api :8002 + analyzer :8001 + MinIO)

## 7. 지금까지 결과

- roi_ocr: 경기 화면 32장에서 35개 영역 모두 정상, 리플레이 화면은 통과 0. 17분 영상 12분 (CPU)
- asr (small, CPU): 17분 영상 13분. 선수명 오류 다수 (케리아 8/8 "캐리아", 티원 → "티언", 페이즈·쉰은 한 번도 못 잡음). 영어 해설도 Xun → "Shun", Bin → "Ben"
- 번역 테스트 (gemini-3.8-flash, 20줄): 고유명사 A 30/31, B 30/31, C 31/31. 유명 선수는 LLM이 이미 알아서 A에서도 고침. 키아나(아이콘뿐)는 C만 맞힘

## 8. 미정

- OCR 결과를 프롬프트에 넣는 범위 (대사 앞뒤 몇 초, 어떤 영역까지)
- ASR 모델 크기 (small → large-v3-turbo), 맥 GPU(mlx) 사용 여부
- 실시간 지연 측정 (로컬 심서버) 여부
- 비교 종목·다른 화면 구성 추가 (과제 3의 "어떤 종목·화면 구성")

## 9. 다음 할 일

- [x] 대사 검수: ASR 원문을 사람이 고치면 고친 문장으로 번역 (ui 검수 화면, api `LineReview`, analyzer `text_overrides`)

1. [ ] 챔피언: 시작 화면 선수 카드 OCR
2. [ ] translate 전체 대사 실행 (A·B → C)
3. [ ] 케이스셋: 대사 ↔ 화면 OCR 시간 맞추기, 이름 오류 후보 자동 추출, 검수 (api `/cases`)
4. [ ] evaluate: 조건별·유형별 집계
5. [ ] 다른 종목·화면 구성 영상 추가
