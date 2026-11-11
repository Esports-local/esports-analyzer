# 작업 체크리스트

## 완료
- [x] preprocess: 영상 정보, VFR 점검, 16kHz wav
- [x] segments: TransNetV2 전환 검출 (cut/gradual + black)
- [x] hls: 전환 기준 ts (≤4초 균등 분할, ts마다 I-frame 1개), m3u8 표시
- [x] asr: VAD 덩어리마다 언어 감지 후 받아쓰기 (합성 음성으로 실행 확인)
- [x] `run_pipeline()`, 미구현 단계 건너뛰기, 단계별 시간·메모리 기록 (`work/run_log.json`)
- [x] ROI 설정 `configs/lol/worlds_2026.json` (팀·순위·킬·골드·타워·시간·선수 이름·KDA, 35개). 스크린샷 프레임에서 전부 정상 인식
- [x] roi_ocr: 화면이 바뀔 때만 OCR, 같은 값이 이어진 구간 저장 (합성 클립으로 확인)

## 실제 영상이 오면 (Worlds 2026 4강 T1 vs BLG 2세트)
- [ ] roi_ocr: 좌표가 실제 프레임과 맞는지, KDA·골드·시간이 바뀔 때 구간이 나뉘는지
- [ ] roi_ocr: 리플레이·캠·밴픽 화면에서 엉뚱한 글자가 검증에서 걸러지는지
- [ ] roi_ocr: 블루가 골드를 앞설 때 gold_diff 블루 좌표, 긴 닉네임 칸 폭
- [ ] roi_ocr: 팀 킬 = 선수 킬 합계 검사
- [ ] asr: 해설 받아쓰기 정확도, 선수·챔피언 이름을 어떻게 받아 적는지. 모델 크기(small → medium/large-v3) 결정
- [ ] segments 검수: 중계 화면 전환(경기 화면 ↔ 캠·리플레이)을 잘 나누는지
- [ ] hls 화질(CRF 20 / 23 / 26) 결정 → `scripts/compare_crf.py`
- [ ] 1시간 영상 처리 시간·메모리 (`work/run_log.json`)

## 번역·평가
- [ ] 번역 모델 결정 (조건 A/B/C 같은 모델)
- [ ] translate: 조건 A(음성만), B(+OCR) → 이어서 C(+영상 프레임). 비교용 규칙 치환 조건 검토
- [ ] 경기 정보: Leaguepedia 세트별 포지션·선수·챔피언 (요청 제한 주의)
- [ ] 한국어 표기 ↔ 영문 연결 (선수·챔피언). 위키백과 API 또는 직접 입력
- [ ] evaluate: 고유명사 정확도, 조건별·종류별·화면 상태별 집계

## 인프라와 연동
- [ ] 큐 + 워커 (`run_pipeline` 호출)
- [ ] api 연동 스펙 정리
- [ ] MinIO 업로드
- [ ] asr 화자 구분: pyannote (필요하면)
