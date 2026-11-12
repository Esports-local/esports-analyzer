"""엔드포인트 공통: 원본 영상·결과 폴더 위치. HTTP 와 무관하게 pipeline·스크립트에서도 쓸 수 있다."""

import os
from pathlib import Path

VIDEOS_ROOT = Path(os.environ.get("VIDEOS_ROOT", "/data")).resolve()
OUTPUTS = Path(os.environ.get("OUTPUTS_ROOT", "outputs")).resolve()  # 띄운 위치와 상관없이 같은 폴더


def video_path(name: str) -> Path:
    """VIDEOS_ROOT 바로 아래 영상 파일.

    돌려주는 경로는 요청한 이름 그대로다 (바로가기를 따라가지 않는다).
    결과 폴더 outputs/<이름에서 확장자 뺀 것>/ 이 api 가 아는 work_name 과 같아야 하기 때문.

    Raises:
        FileNotFoundError: 파일이 없거나 VIDEOS_ROOT 바로 아래가 아니다.
    """
    path = VIDEOS_ROOT / name
    if path.parent.resolve() != VIDEOS_ROOT or Path(name).name != name or not path.is_file():
        raise FileNotFoundError(f"영상이 없습니다: {name}")
    return path
