# 작업 체크리스트

## 완료
- [x] preprocess: 영상 정보, VFR 점검, 16kHz wav
- [x] segments: TransNetV2 전환 검출 (cut/gradual + black)
- [x] hls: 전환 기준 ts (≤4초 균등 분할, ts마다 I-frame 1개), m3u8 표시
- [x] faces: InsightFace 얼굴 군집 F1, F2 …
- [x] `run_pipeline()`, 미구현 단계 건너뛰기, 단계별 시간·메모리 기록 (`work/run_log.json`)
- [x] 코드리뷰 1~5번 수정

## 실제 영상이 오면
- [ ] segments 검수: 컷 놓침 / 가짜 컷, 와이프, 밤 장면
- [ ] faces 검수: 유사도 기준 0.45, 작은 얼굴
- [ ] 리뷰 3번: ffmpeg 실패 처리
- [ ] 리뷰 4번: 1시간 영상 메모리
- [ ] 리뷰 5번: 얼굴이 많을 때 군집화 속도
- [ ] 리뷰 6번: hls 처리량 (ts마다 ffprobe 실행)
- [ ] 리뷰 7번: fps가 두 배로 잡히는지
- [ ] 리뷰 8번: 영상 시작 시각 어긋남
- [ ] 리뷰 9번: 회전 정보, 영상 정보 중복 조회
- [ ] hls 화질(CRF 20 / 23 / 26) 결정 → `scripts/compare_crf.py`

## 파이프라인 단계 (코드 작성 완료, 워크스테이션에서 빌드·실행 검증 필요)
- [x] asr: VAD 덩어리마다 언어 감지 후 받아쓰기 (합성 음성 한·일 대사로 실행 확인, 2026-11-11)
  - [ ] 실제 대사로 받아쓰기 정확도·`language_min_prob`(0.7) 확인, 모델 크기(small → medium/large-v3) 결정
- [ ] asr 화자 구분: pyannote 연결 코드만 있음 → HF_TOKEN, 버전 고정, 실행 검증
- [x] similarity: open_clip 화면 유사도 + 대사 연속성 (합성 영상으로 실행 확인)
  - [ ] MobileCLIP 교체 여부. 같은 사진의 다른 구도끼리도 visual 0.73~0.75 라 기준값 0.80 확인 필요
- [x] scenes: 초기 규칙 (합성 영상에서 의도한 3씬으로 나뉨)
  - [ ] 실제 영상으로 기준값 조정
- [x] vlm: mock 백엔드 실행 확인
  - [ ] GPU 백엔드(Qwen3-VL) 구현 (워크스테이션)
- [x] episode: 최종 `analysis/episode.json` 실행 확인 (화자 요약은 화자 구분이 꺼져 있어 비어 있음)
- [ ] 이미지 빌드: faster-whisper, open-clip-torch 설치 확인, 모델 캐시(HF_HOME) 볼륨
- [x] api·ui 개발용 샘플 결과물: `scripts/make_tts_clip.py` 로 만든 60초 합성 클립 → `outputs/tts_drama/` (`hls/`, `analysis/episode.json`)

## 인물 연결
- [ ] 화자 ↔ 얼굴 연결 (Light-ASD)
- [ ] 얼굴 ↔ 배우 사진 비교 (TMDB)

## api 쪽에서 할 것 (main 아님, 참고용)
- 예측: 1차 LLM 호출(header만) → 요청 JSON(회차·인물·키워드·위치) → 코드에서 body 선별 → 2차 호출(예측 + 근거 씬)
- 1차 출력 스키마, 근거 씬 검증 (2차에 넘긴 씬 중 하나인지)
- 예측 모델: Gemini 3.6 Flash / 3.5 Flash-Lite 중 결정, 30초 이내

## 인프라와 연동
- [ ] 큐 + 워커 (`run_pipeline` 호출)
- [ ] api 연동 스펙 정리
- [ ] MinIO 업로드 (`hls/`, `analysis/`)
- [ ] GPU 서버용 Dockerfile (VLM, NVENC)

## 라벨링 확인 웹 (별도 레포)
- [ ] 영상별 라벨링(ts·segment·scene·대사·얼굴)이 잘 됐는지 확인하는 웹

## 정리
- [ ] OrbStack 파일 공유 문제 원인 확인
- [ ] `../PLAN.md`에 지금까지 정한 내용 반영
