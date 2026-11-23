# esports-analyzer — 분석 파이프라인 (FastAPI)

> 프로젝트 전체 구성, 결정 사항, 저장소 구조는 **`docs/PLAN.md`가 기준**이다 (api·ui 레포도 이 파일을 본다). 작업 전에 반드시 읽는다.
> 주제: LoL 중계 영상 → 경기 정보(header) + 화면 OCR + 해설 받아쓰기로 시간별 경기 데이터를 만들고, 그걸로 숏폼을 만든다.
> 드라마 스토리 예측 버전(faces·similarity·scenes·vlm·episode 단계)은 `drama` 브랜치에 있다.
> api ↔ analyzer 연동 스펙은 `api/<엔드포인트>/schemas.py`와 docs/PLAN.md 6장이 기준이다 (`/videos`, `/jobs`, `/shorts`). 바꾸면 esports-api 의 `api/games/analyzer.py`도 같이 고친다.

## 스킬 먼저 (esports-dev 플러그인)
아래 작업은 코드를 보거나 명령을 치기 전에 해당 스킬을 먼저 부른다. 이 문서의 구조·명령만 보고 바로 하면 스킬에 적힌 순서와 같이 고칠 곳(스펙, 다른 레포)을 빠뜨린다.
- 엔드포인트 추가, `/videos`·`/jobs` 요청·응답 모양 변경(파라미터·필드 추가 포함) → `esports-dev:add-endpoint`
- 로컬 서버 띄우기·끄기·로그 → `esports-dev:local-up`

## 이 레포의 역할
중계 영상 분석 파이프라인. HTTP 서버는 작업 접수/조회/취소만 하는 얇은 층이고, 실제 처리는 `pipeline/`에서 한다.
분석은 수십 분 걸리는 작업이므로 FastAPI BackgroundTasks가 아니라 큐 + 워커로 실행할 예정.

## 구조
- 테스트 `tests/` (HTTP 서버 스모크, 분석 단계는 안 돌림). CI: `.github/workflows/ci.yml` (푸시·PR 마다 문법 확인 + 테스트, faster-whisper 등 무거운 의존성은 설치 안 함)
- `pipeline/` 폴더: 실행(`stages.py` · `__main__.py`) · 공통(`io.py` 결과 저장, `schema.py`, `media.py` ffprobe·ffmpeg 프레임 읽기) · `header/`(경기 정보) · `screen/`(레이아웃·화면 OCR·장면 전환·화면 종류) · `audio/`(전처리·받아쓰기) · `game/`(이벤트, 다음: 바디 context) · `shorts/`(후보 장면·렌더, 다음: LLM 순위·제목)
- `pipeline/stages.py`: 단계 목록과 `run_stage`, `run_pipeline` (코드에서 실행: `from pipeline import run_pipeline`). 각 단계는 `outputs/<영상명>/work/<stage>.json` 출력 → 중간부터 재실행 가능
  - 단계: header → preprocess → segments → roi_ocr → asr → timeline → context(바디) → shorts(숏폼 후보) → evaluate(예정). hls(재생용 ts)·translate(번역)는 지웠다 (git 기록에 있음)
- `pipeline/header/`: 경기 기본 정보 `work/header.json` (영상 안 읽음, 설정 `match`(op.gg 경기 번호)·`set` 필요, api 가 영상 등록 때 넘김). `pipeline/header/opgg.py`(경기 페이지 `__NEXT_DATA__` + 세트 정보 `/matches/graphql` GetGameByMatch 하나, httpx)와 `pipeline/header/ddragon.py`(챔피언 번호 → 영어·한국어 이름·스킬, 경기 패치 버전)로만 만든다 (사람·LLM 입력 없음). 경기·팀·선수·챔피언·스킬·밴·세트 결과, 시리즈 스코어·앞 세트 밴픽·피어리스로 못 쓰는 챔피언(`FEARLESS_LEAGUES` 2025년 이후). 같은 요청으로 `work/opgg_timeline.json`(이 세트 기록: 오브젝트 종류·시각·막타, 킬한 선수·좌표, 1분 위치·골드, 경기 시각 기준)도 만든다. 킬·오브젝트 이벤트는 이걸로 만든다 (11-23 결정: VOD 라 op.gg 기록이 이미 있으므로 화면으로 다시 만들지 않는다. timeline 참고). op.gg 가 안 되면 `error`만 남기고 작업은 계속 (timeline 이 화면 숫자로 대신)
- `pipeline/screen/segments.py`: TransNetV2(원래 fps, 10fps 로 줄이면 하드컷을 절반 놓침)로 장면 전환 cut · dissolve · wipe → segment. 리플레이 들어가고 나올 때 로고 그래픽 전환(컷 두 개 + 사이가 계속 움직임)은 wipe 하나로 합친다. 레이아웃 파일에 화면 종류 규칙 `screens`가 있으면(지금은 LCK) `pipeline/screen/rules.py`가 2초마다 정해진 자리 글자(경기 시간, REPLAY/HIGHLIGHT, GAME TIME, 세트 사이 카운트다운 시계, 밴픽 막대 PATCH)를 RapidOCR 로 읽어 segment 마다 다수결로 `kind`(live · replay · postgame · countdown · draft · break)와 경기 구간 `game`을 붙인다. 경기 끝은 header 경기 길이로 맞추고 postgame 화면의 경기 길이·승자로 확인(`checks`). 56분 영상에 TransNetV2 약 25분(CPU 69장/초)
- `pipeline/screen/roi_ocr.py`: 중계 화면 고정 영역 OCR (RapidOCR 글자 인식만). 레이아웃 파일 `configs/<종목>/<대회>.json` 하나에 `rois`(영역: `ocr_roi` [x1,y1,x2,y2] 1920x1080 기준, `char_map`(글꼴 오인식 바꾸기, LCK 숫자 0 → o), `allowed_text`, `validation`, `static`)와 `screens`(segments 화면 종류 규칙, 선택)가 같이 있다. 설정 `layout` 기본 `auto`: `pipeline/screen/layout.py`가 header 리그로 `configs/lol/layouts.json`에서 고른다 (LCK → `lol/lck_2026.json`(치지직 중계, 선수 이름은 옆 패널 "GEN Kiin", 순위 칸 없음), 그 밖·header 없음 → `lol/worlds_2026.json`). 직접 주면 그대로 씀. 새 대회는 `configs/lol/_template.json`을 복사해 좌표를 채우고 `python -m pipeline.screen.layout <영상> <초> --layout <파일> --out <png>`(영역·읽은 글자·화면 종류를 그린 그림)로 맞춘 뒤 `layouts.json`에 리그 한 줄 추가.
  건너뛰는 영역: `skip_tags`(항상, gold_diff: 쓰는 곳 없음) · `opgg_skip_tags`(op.gg 상세 기록이 있을 때, banner·dragon_icons·obj_count). 10분 구간 152초 → 72초. 화면 OCR 로 남기는 이유: 경기 시간(op.gg 시각 → 영상 초 동기화, 필수) · KDA(킬당한 선수·도움, op.gg 에 없음) · 타워(op.gg 에 시각 없음) · 팀 킬 수(동기화 검증, op.gg 없을 때 대체) · 팀 골드(1초 단위라 싸움 전후 변화가 잡힘. op.gg 는 1분 간격)
  결과 `work/roi_ocr.json`: 영역(tag+slot)마다 같은 값이 이어진 구간. 화면이 그대로면 OCR 을 다시 돌리지 않는다. segments 구간 종류가 replay·postgame·draft·countdown·break 면 OCR 을 건너뛴다 (`skip_kinds`, asr 도 draft·countdown·break 구간은 무음 처리해 받아 적지 않음)
- `pipeline/audio/asr.py`: faster-whisper (Silero VAD 내장, 대사 단위 언어 감지). CPU 면 int8 (`device: auto`도 GPU 가 없으면 int8. 전에는 float32 로 돌았다: 10분 구간 351초 → 187초, 메모리 2.7GB → 1.4GB). BatchedInferencePipeline(int8 에서 9% 더 빠르지만 글자 차이가 두 배)과 roi_ocr·asr 동시 실행(CPU 를 서로 뺏어 10% 느림)은 재 보고 뺐다. 해설 대사 → `work/asr.json`. 끝에 `pipeline/audio/terms.py`가 header 단어(이 경기 챔피언 → 설정 `terms_extra`로 받는 선수·팀 한국어 이름 → 밴·스킬 순)로 고유명사를 고친다 (한글 자모 유사도, 후보가 하나이고 충분히 비슷할 때만, 조사 받침 맞춤). 원문 `raw`, 바꾼 내역 `fixes`. 이미 있는 asr.json 에 다시 적용: `python -m pipeline.audio.terms outputs/<영상>/work`
- `pipeline/game/timeline.py`: 경기 이벤트 `work/timeline.json` (LLM 없음). **op.gg 상세 기록이 있으면 킬·오브젝트는 op.gg** (경기 시각 → 영상 초는 화면 시계 `context.build_clock`·`to_video`, 리플레이 중이면 60초 안 가까운 시계 구간으로. killer 는 op.gg, victims·assists 는 화면 KDA 변화), 타워는 항상 화면 숫자. 동기화 검증 `sync`: op.gg 킬 ↔ 화면 팀 킬 수가 오른 시각 차이(`ocr_lag_sec`, 정상 1~2초, -0.5~4초 밖이면 outliers). 결과 `source`(opgg·ocr)·`unmapped`. op.gg 가 없으면 아래 화면 숫자 방식: 화면 숫자(팀 킬·타워·오브젝트 수)가 오른 시점. LCK 는 점수판 아래 띠의 오브젝트 배너(OCEAN DRAKE · BARON NASHOR …, roi `banner`)로 용·바론·전령 종류를 잡고, 용은 경기 시간 옆 용 아이콘 픽셀 비율(roi `dragon_icons`, `measure: fill`)이 늘어난 진영을 팀으로 본다. obj_count 의 오브젝트 종류는 레이아웃 `subtype`(LCK grubs). 잡음 정리(min_samples, 줄어든 값·급증 버림), 못 읽은 구간이 끼면 start_sec~end_sec 범위로, 킬은 앞뒤 3초 KDA 변화로 killer·victims·assists, 앞뒤 5초 대사 id
- `pipeline/game/context.py`: 바디 `work/context.json` (LLM 없음, 이벤트는 timeline 그대로라 op.gg 기록이 들어 있다). game(경기 구간·승자) · players(슬롯→선수·챔피언) · clock(화면 경기 시간으로 영상 초↔경기 초, 리플레이·멈춤·거꾸로 간 오독은 끊거나 버림) · segments(컷 사이 구간 kind·들어온 전환) · events(timeline 이벤트 + 슬롯·first_blood·multikill·teamfight 태그·화면 종류) · signals(초 단위 소리 크기·골드 차이·컷 빈도·화면 종류). 하이라이트 고르기는 shorts 쪽
- `pipeline/game/bundle.py`: 분석이 끝날 때마다 `outputs/<영상>/analyzer.json` (sevenmin_master `mastering_data.json` 방식). header(경기 지식·시계·segment_types) · segments(시간순 트리: phase(game·draft·postgame·countdown·break) → cut(장면, 그 안의 events·lines·끝 경기 상태)) · header 에 `event_source`·`sync`(timeline 동기화 검증) · 장면 안에 events(앞뒤 대사 문장, op.gg 위치가 있으면 맵 구역 area)·lines·signals·state(장면 끝 팀 킬: op.gg 누적, 골드 차이: 화면 값 먼저·없으면 op.gg 1분 보간, gold_source). op.gg 기록 원본·OCR 원본·run_log 는 넣지 않음. 예: `docs/examples/lck_final_g1_900-1500s/analyzer.json`
- `pipeline/shorts/scenes.py`: 단계 shorts → `work/shorts.json` (LLM 없음). 컷 경계를 넘는 대사 문장이 있으면 앞뒤 컷을 한 장면으로 묶고(화면 종류가 바뀌면 나눔, `cross_tol_sec` 0.3), 킬·오브젝트(유충 제외)가 있는 live 장면을 후보로. 길이는 자르지 않음. 이벤트별 클립 `clips`(모음형 재료): 이벤트가 든 장면 안에서 앞뒤 문장 중 소리 0.85 이상인 것까지(4~12초 앞, 3~8초 뒤, 문장 경계), 뒤 90초 안 리플레이 장면 `replay`. 컷 트리는 그때 `bundle.build`로 만든다. 표 보기 `python -m pipeline.shorts.scenes outputs/<영상>`
- `pipeline/shorts/rank.py`: 장면형 LLM (shorts 단계 안, 경기당 1번, `pipeline/llm.py` Gemini REST·응답 스키마 지정, 모델 `gemini-3.8-flash`). 입력 `llm_input.build` → 후보마다 {rank, start_line, end_line, titles 3개(공백 포함 10자 이내), reason} → 코드 검사(고침: lines_in_scene·covers_events(2초 여유)·length(8초)·title_rules / 표시: 180초 넘음·전투 도중 시작·끝(앞뒤 문장 loud 0.9)·title_names·title_counts·rank_vs_rule) → 후보에 rank·title·titles·reason·clip_start/end_sec·source(llm·llm+fixed·rule)·warnings. 기록 `work/shorts_llm.json`(입력 해시·원 응답·토큰·checks·final), 입력 `work/shorts_llm_input.json`. 같은 입력·모델·프롬프트 버전이면 다시 부르지 않음. 키가 없거나 실패하면 규칙 점수 순서·장면 전체·자동 제목. 설정 `use_llm`·`llm_model`. 예시 구간 1번 호출: 입력 8.9k·생각 2.5k·출력 0.6k 토큰
- `pipeline/shorts/query.py`: 모음형 요청 글 → 조건 {killer 슬롯, type, team, tag} + 10자 제목 + note (LLM 은 요청 글과 선수 목록만 봄, 허용된 값만 남김). `POST /shorts/query`
- `pipeline/shorts/render.py`: 숏폼 렌더 (ffmpeg 한 번, 구간 개수·길이 제한 없음). 720x1280 검은 화면(11-23 참고 쇼츠 구성): 위 제목 1~2줄(`title_lines`, 첫 줄 흰색·둘째 줄 노란색) · 가운데 1920x1080 중계 화면 가운데 1180x1080 을 잘라 720x658 (y=320, 점수판·KDA 표가 보임) · 게임 화면 아래쪽 설명 자막(`captions`, 원본 시각 → 이어 붙인 영상 시각, `output_spans`). 나눔스퀘어 Bold + 검은 테두리. BGM(`pipeline/shorts/bgm.py`가 `assets/bgm/`에서 key 로 고른 곡, 곡 파일은 git 제외)이 있으면 반복해 깔고 해설이 나오면 줄임(sidechaincompress), 렌더 기록에 bgm. 임시 파일에 쓰고 이름을 바꿔 결과로 만든다, 옆에 렌더 기록 .json
- `pipeline/__main__.py`: CLI. `--set 단계.키=값`으로 설정 덮어쓰기 (예: `--set header.match=32655`). `--start-sec 900 --max-sec 300`: 900~1200초만 분석(빠른 확인용, 결과 시각은 원본 기준), 결과는 `outputs/<영상명>_900-1200s/`에 따로 (header 도 그 폴더에 다시 만들어야 레이아웃 자동 선택이 된다)
- `app/main.py`: FastAPI 앱. 엔드포인트 router 를 묶기만 한다 (`uvicorn app.main:app`)
- `api/`: HTTP 엔드포인트. 엔드포인트마다 폴더 하나 (`views.py` router + `schemas.py` 요청·응답 모양 = 연동 스펙). 큐는 아직 없음
  - 공통: `views.py`(`GET /health`), `paths.py`(`VIDEOS_ROOT`·`OUTPUTS_ROOT` 절대 경로, 영상 경로 확인. 결과 폴더 이름은 요청한 파일 이름 기준이라 바로가기여도 api 의 work_name 과 같다)
  - `videos/`: `GET /videos` (VIDEOS_ROOT 영상, 1920x1080·소리 있는 것만 selectable), `GET /videos/{name}/file` (원본 영상 그대로, Range 요청 가능, 허용 출처는 ui 주소 하나 `VIDEO_ALLOW_ORIGIN`(기본 `http://localhost:8000`), compose 는 포트를 127.0.0.1 에만 연다. api 가 `video_url`로 ui 에 넘겨 컷 탭에서 재생·프레임 뜨기)
  - `jobs/`: `POST /jobs`(파이프라인을 프로세스로 실행, 한 번에 하나, 로그 `work/job.log`), `GET /jobs/{id}`, `POST /jobs/{id}/cancel`
    - 같은 job_id 를 다른 내용으로 보내면 409. 취소는 프로세스 그룹째(ffmpeg 등 하위 프로세스까지). 끝난 작업 기록은 최근 100개만 남김
  - `shorts/`: `POST /shorts`(숏폼 렌더 접수, `title_lines`·`captions`는 api 가 이벤트로 만들어 보냄, 상태에 `bgm`), `GET /shorts/{key}`, `POST /shorts/query`(모음형 요청 글 → 조건, LLM. 못 쓰면 error 만). /jobs 와 달리 거절하지 않고 줄을 세운다 (워커 스레드 하나). 같은 key·내용이면 기존 상태, 실패한 것이면 다시 줄 세움, 다른 내용이면 409. 서버를 다시 띄워도 결과 파일이 있으면 succeeded

```
docker build -t esports-analyzer .
docker run --rm -p 8001:8001 -v "$PWD":/app esports-analyzer uvicorn app.main:app --host 0.0.0.0 --port 8001 --reload
docker run --rm -v "$PWD":/app -v ~/vod-data:/data esports-analyzer python -m pipeline /data/game.mp4 --only preprocess,roi_ocr,asr
```
현재 Dockerfile은 CPU용. GPU가 필요해지면 CUDA 베이스 이미지로 별도 작성.

## 모델
TransNetV2(장면 전환, CPU) · RapidOCR(글자 인식, CPU: 화면 OCR · 화면 종류) · faster-whisper(ASR, Silero VAD 내장) · pyannote Community-1(화자 구분, 선택, HF_TOKEN)

## 규칙 요약 (상세는 docs/PLAN.md)
- 시간은 초 단위 float, 단계 결과 JSON에는 `schema_version`
- `work/`: 단계별 중간 결과 (외부 공개 안 함) · `shorts/`: 숏폼 렌더 결과
- 화면 좌표는 1920x1080 기준 픽셀. 다른 해상도 영상은 그 크기로 맞춰 읽는다
- 패키지는 `requirements.txt`에 `==` 고정 후 이미지 재빌드 (로컬 가상환경 안 씀)
- 중계 영상은 연구용으로 로컬에서만 쓰고 재배포하지 않는다 (`*.mp4`, `data/`, `outputs/`는 git 제외)

## 커밋 규칙
- 메시지는 `종류/내용` 한 줄로 간단하게 쓴다
  - `feat/` 기능 구현 · `refactor/` 구조 변경 · `fix/` 버그 수정 · `docs/` 문서
  - 예: `feat/TransNetV2 segments 단계 구현`
- 작성자는 사용자 본인만. Claude 를 `Co-Authored-By` 등 공동 작성자로 넣지 않는다
