"""op.gg 경기 페이지와 세트 정보 읽기 (header 단계가 쓴다).

- 경기: esports.op.gg/matches/<번호> 페이지는 경기 정보를 페이지 안 JSON(__NEXT_DATA__)으로 같이 보낸다.
  robots.txt 는 /matches 를 막지 않는다.
- 세트별 진영·선수·챔피언·밴·승자는 그 JSON 에 없고, 페이지가 뜬 뒤 op.gg 화면이 /matches/graphql 의
  GetGameByMatch(matchId, set) 로 불러온다. 같은 쿼리에서 필요한 필드만 골라 세트당 한 번 요청한다
  (op.gg 내부 API 는 이 하나만 쓴다).
- 요청은 프로세스 안에서 1초에 한 번만 보낸다 (_throttle). 많이 보내면 op.gg 가 429 로 막는다.
"""

import json
import re
import threading
import time

import httpx

BASE = "https://esports.op.gg"
USER_AGENT = "esports-analyzer/0.1 (research; match metadata)"
TIMEOUT = 10
_NEXT_DATA = re.compile(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.S)
GRAPHQL_URL = f"{BASE}/matches/graphql"
GAME_QUERY = """query GetGameByMatch($matchId: ID!, $set: Int) {
  gameByMatch(matchId: $matchId, set: $set) {
    id
    length
    winner { id }
    teams { side bans team { id acronym } }
    players { side position championId player { nickName } }
  }
}"""
MIN_INTERVAL = 1.0  # 초, op.gg 요청 사이 최소 간격
_lock = threading.Lock()
_last = 0.0
ROLES = {"top": "top", "jun": "jungle", "mid": "mid", "adc": "bot", "sup": "support"}  # op.gg → 포지션 이름


def _throttle() -> None:
    """앞 요청에서 MIN_INTERVAL 초가 지날 때까지 기다린다 (스레드가 여럿이어도 한 줄로)."""
    global _last
    with _lock:
        wait = _last + MIN_INTERVAL - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last = time.monotonic()


def match_url(ref: str | int) -> str:
    """경기 번호나 주소를 받아 페이지 주소로 바꾼다.

    Raises:
        ValueError: 번호도 op.gg 경기 주소도 아님.
    """
    ref = str(ref).strip()
    if ref.isdigit():
        return f"{BASE}/matches/{ref}"
    if re.match(r"^https://esports\.op\.gg/(?:[a-z]{2}/)?matches/\d+", ref):
        return ref
    raise ValueError(f"op.gg 경기 번호나 주소가 아닙니다: {ref}")


def fetch_match(ref: str | int) -> tuple[str, dict]:
    """페이지를 받아 (주소, match JSON).

    Raises:
        httpx.HTTPError: 요청 실패.
        ValueError: 페이지에 경기 데이터가 없음.
    """
    url = match_url(ref)
    _throttle()
    res = httpx.get(url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT, follow_redirects=True)
    res.raise_for_status()
    return str(res.url), parse_match(res.text)


def parse_match(html: str) -> dict:
    """페이지 HTML 에서 match JSON 을 꺼낸다. Raises: ValueError."""
    m = _NEXT_DATA.search(html)
    if not m:
        raise ValueError("페이지에 __NEXT_DATA__ 가 없습니다")
    match = json.loads(m.group(1)).get("props", {}).get("pageProps", {}).get("match")
    if not match:
        raise ValueError("페이지에 경기 데이터(match)가 없습니다")
    return match


def set_videos(match: dict) -> dict[int, str]:
    """{세트 번호: 전체 영상 주소}. op.gg 는 세트 정보를 영상에 붙여서만 준다."""
    out: dict[int, str] = {}
    for v in match.get("videos") or []:
        g = v.get("game") or {}
        if g.get("set") and v.get("type") == "full-game" and v.get("url"):
            out[int(g["set"])] = v["url"]
    return out


def fetch_game(match_id: int | str, set_number: int) -> dict | None:
    """세트 하나의 gameByMatch JSON. 열리지 않은 세트는 None.

    Raises:
        httpx.HTTPError: 요청 실패.
        ValueError: GraphQL 오류 응답.
    """
    _throttle()
    res = httpx.post(
        GRAPHQL_URL, timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Referer": f"{BASE}/matches/{match_id}"},
        json={"operationName": "GetGameByMatch", "query": GAME_QUERY,
              "variables": {"matchId": str(match_id), "set": set_number}})
    res.raise_for_status()
    body = res.json()
    if body.get("errors"):
        raise ValueError(f"op.gg 세트 조회 오류: {body['errors'][0].get('message')}")
    return (body.get("data") or {}).get("gameByMatch")


def parse_game(game: dict) -> dict:
    """gameByMatch JSON → {opgg_game_id, length_sec, winner: 팀 op.gg id | None, sides: {진영: 팀 op.gg id},
    bans: {진영: [챔피언 번호]}, players: [{side, role, name, champion_id}]}. 모르는 포지션의 선수는 빠진다.

    Raises:
        KeyError, TypeError, ValueError: 응답 모양이 다름.
    """
    teams = [t for t in game.get("teams") or [] if t.get("side") in ("blue", "red")]
    players = [{"side": p["side"], "role": ROLES[p.get("position")], "name": (p.get("player") or {}).get("nickName") or "",
                "champion_id": str(p.get("championId") or "")}
               for p in game.get("players") or [] if p.get("side") in ("blue", "red") and p.get("position") in ROLES]
    winner = (game.get("winner") or {}).get("id")
    return {"opgg_game_id": int(game["id"]), "length_sec": game.get("length"),
            "winner": int(winner) if winner else None,
            "sides": {t["side"]: int(t["team"]["id"]) for t in teams if (t.get("team") or {}).get("id")},
            "bans": {t["side"]: [str(b) for b in t.get("bans") or []] for t in teams},
            "players": players}
