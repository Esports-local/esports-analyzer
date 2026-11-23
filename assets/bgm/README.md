# 숏폼 BGM

여기에 넣은 곡을 숏폼 렌더 때 해설 아래에 작게 깐다 (해설이 나오면 자동으로 줄어듦). 곡 파일은 git 에 올리지 않는다.

- 라이선스를 확인한 곡만 넣는다. YouTube 오디오 보관함(YouTube Studio) 곡을 추천 (YouTube 업로드에 안전, 대부분 출처 표기 불필요).
- `bgm.json`(아래 형식)이 있으면 그 목록만 쓴다. 없으면 폴더의 mp3·m4a·wav·ogg·flac 을 전부 쓴다.
- 같은 숏폼(key)은 항상 같은 곡. 곡을 추가·삭제하면 고르는 곡이 바뀔 수 있다.
- 이미 렌더한 숏폼에 BGM 을 새로 넣으려면 api `LAYOUT_VERSION`을 올려 다시 렌더한다.

```json
[
  {"file": "track1.mp3", "title": "곡 이름", "source": "YouTube Audio Library", "license": "YouTube Audio Library",
   "attribution": ""}
]
```
