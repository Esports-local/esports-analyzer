"""analyzer HTTP 서버. api 가 영상 목록을 보고, 분석 작업을 맡기고, 상태를 묻는다.

엔드포인트는 `api/<엔드포인트>/views.py` (요청·응답 모양은 같은 폴더의 schemas.py). 여기서는 묶기만 한다.
"""

from fastapi import FastAPI

from api import views as common
from api.jobs import views as jobs
from api.shorts import views as shorts
from api.videos import views as videos

app = FastAPI(title="esports-analyzer")

app.include_router(common.router, tags=["health"])
app.include_router(videos.router, tags=["videos"])  # 원본 영상 목록
app.include_router(jobs.router, tags=["jobs"])  # 분석 작업 시작·조회·취소
app.include_router(shorts.router, tags=["shorts"])  # 숏폼 렌더 접수·조회 (줄 세움)
