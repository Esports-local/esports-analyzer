# 작업 체크리스트

## 완료
- [x] preprocess: 영상 정보, VFR 점검, 16kHz wav
- [x] segments: TransNetV2 전환 검출 (cut/gradual + black)
- [x] hls: 전환 기준 ts (≤4초 균등 분할, ts마다 I-frame 1개), m3u8 표시
- [x] asr: VAD 덩어리마다 언어 감지 후 받아쓰기. 결과에서 단어별 시간(words)은 뺌
- [x] `run_pipeline()`, 미구현 단계 건너뛰기, 단계별 시간·메모리 기록 (`work/run_log.json`)
- [x] ROI 설정 `configs/lol/worlds_2026.json` (팀·순위·킬·골드·타워·시간·선수 이름·KDA, 35개)
- [x] roi_ocr: 화면이 바뀔 때만 OCR, 같은 값이 이어진 구간 저장
- [x] 실제 영상 (Worlds 2026 4강 BLG vs T1 2세트, 한국어 17분): 경기 화면 32장 35개 영역 정상, 리플레이 통과 0, 킬·KDA 변화 시점 일치
- [x] asr 실제 영상: 한국어 17분 / 영어 10분. 선수명 오류 사례 확인 (docs/PLAN.md 7장)
- [x] 번역 작은 테스트: gemini-3.8-flash, 대사 20줄 × 조건 A/B/C

## roi_ocr
- [ ] `static` 확정(5초) 실제 영상으로 확인
- [ ] 1~2장짜리 잡음 구간 처리 (api 에서는 min_samples 로 거름)
- [ ] 챔피언: 시작 화면 선수 카드 OCR ("KERIA / Cafe Cuties Poppy")
- [ ] 죽음 판정: 초상화 흑백·부활 시간, KDA 데스 변화
- [ ] 리플레이 판별 템플릿 (OCR 건너뛰어 속도 개선)

## asr
- [ ] 모델 크기 (small → large-v3-turbo), 맥 GPU(mlx) 비교
- [ ] 화자 구분: pyannote (필요하면)

## 번역·평가
- [x] translate 단계 구현 (A/B 기본, C 선택, dry_run, 재사용)
- [ ] translate 전체 대사 실행 (A·B, 이어서 C)
- [ ] 케이스셋: 대사 ↔ 화면 OCR 시간 맞추기, 이름 오류 후보 자동 추출
- [ ] evaluate: 고유명사 정확도, 조건별·종류별·화면 상태별 집계

## 인프라와 연동
- [ ] 큐 + 워커 (`run_pipeline` 호출). 지금은 api `import_outputs`로 결과 폴더를 넣음
- [ ] MinIO 업로드
