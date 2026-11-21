"""op.gg 경기 페이지와 세트 정보 읽기 (header 단계가 쓴다).

- 경기: esports.op.gg/matches/<번호> 페이지는 경기 정보를 페이지 안 JSON(__NEXT_DATA__)으로 같이 보낸다.
  robots.txt 는 /matches 를 막지 않는다.
- 세트별 진영·선수·챔피언·밴·승자는 그 JSON 에 없고, 페이지가 뜬 뒤 op.gg 화면이 /matches/graphql 의
  GetGameByMatch(matchId, set) 로 불러온다. 같은 쿼리에서 필요한 필드만 골라 세트당 한 번 요청한다
  (op.gg 내부 API 는 이 하나만 쓴다).
- 이 세트는 같은 요청에 세트 기록 필드(오브젝트·킬·1분 위치·1분 골드)를 붙여 받는다 → parse_timeline
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
GAME_FIELDS = """id
    length
    winner { id }
    teams { side bans team { id acronym } }
    players { side position championId player { nickName } }"""
# 이 세트 기록 (opgg_timeline.json). op.gg 화면의 CoreGame 조각에서 고른 필드. 같은 요청에 붙여 보낸다
TIMELINE_FIELDS = """detailedStats
    teams { side towerKills inhibitorKills dragonKills elderDrakeKills baronKills heraldKills atakhanKills
            frames { gold timestamp } }
    objectKills { teamId objectType objectSubType timestamp }
    players { side position events { type timestamp position { x y } monsterType } frames { position { x y } timestamp } }"""
GAME_QUERY = """query GetGameByMatch($matchId: ID!, $set: Int) {
  gameByMatch(matchId: $matchId, set: $set) {
    %s
  }
}"""
OBJECTS = {"HORDE": "grubs", "DRAGON": "dragon", "RIFTHERALD": "herald", "BARON_NASHOR": "baron",
           "ELDER_DRAGON": "elder", "ATAKHAN": "atakhan"}  # op.gg objectType → 이름
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


def fetch_game(match_id: int | str, set_number: int, timeline: bool = False) -> dict | None:
    """세트 하나의 gameByMatch JSON. 열리지 않은 세트는 None. timeline=True 면 세트 기록 필드도 같이 받는다.

    Raises:
        httpx.HTTPError: 요청 실패.
        ValueError: GraphQL 오류 응답.
    """
    _throttle()
    res = httpx.post(
        GRAPHQL_URL, timeout=TIMEOUT,
        headers={"User-Agent": USER_AGENT, "Referer": f"{BASE}/matches/{match_id}"},
        json={"operationName": "GetGameByMatch",
              "query": GAME_QUERY % (GAME_FIELDS + ("\n    " + TIMELINE_FIELDS if timeline else "")),
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


def _clock(sec: float) -> str:
    return f"{int(sec // 60):02d}:{int(sec % 60):02d}"


def parse_timeline(game: dict) -> dict:
    """gameByMatch JSON(timeline=True) → 세트 기록. 시각은 모두 경기 시각(경기 시작부터 초), 좌표는 맵 좌표 (0~약 15000).

    Returns:
        {detailed, length_sec,
         objectives: [{t, game_time, type: grubs|dragon|herald|baron|elder|atakhan, subtype(용 종류), side, killer(슬롯), position}],
         kills: [{t, game_time, side, killer(슬롯), position}]   죽은 선수·어시스트는 op.gg 에 없다
         positions: {슬롯: [[t, x, y], ...]}  1분마다,  gold: {진영: [[t, 골드], ...]}  1분마다,
         totals: {진영: {towers, inhibitors, dragons, elders, barons, heralds, atakhans}}}
        슬롯은 진영_포지션 (blue_top …).

    Raises:
        KeyError, TypeError, ValueError: 응답 모양이 다름.
    """
    sides = {str(t["team"]["id"]): t["side"] for t in game.get("teams") or []
             if t.get("side") in ("blue", "red") and (t.get("team") or {}).get("id")}
    slot = lambda p: f"{p['side']}_{ROLES[p['position']]}"
    players = [p for p in game.get("players") or [] if p.get("side") in ("blue", "red") and p.get("position") in ROLES]
    monster_events = [(slot(p), e) for p in players for e in p.get("events") or [] if e.get("type") == "ELITE_MONSTER_KILL"]

    objectives = []
    for o in sorted(game.get("objectKills") or [], key=lambda o: o["timestamp"]):
        t = o["timestamp"] / 1000
        who = next(((s, e) for s, e in monster_events
                    if e.get("monsterType") == o.get("objectType") and abs(e["timestamp"] - o["timestamp"]) <= 1000), None)
        sub = (o.get("objectSubType") or "").removesuffix("_DRAGON").lower() or None
        objectives.append({"t": round(t, 3), "game_time": _clock(t), "type": OBJECTS.get(o.get("objectType"), str(o.get("objectType")).lower()),
                           "subtype": sub, "side": sides.get(str(o.get("teamId"))),
                           "killer": who[0] if who else None, "position": who[1].get("position") if who else None})
    kills = sorted(({"t": round(e["timestamp"] / 1000, 3), "game_time": _clock(e["timestamp"] / 1000), "side": p["side"],
                     "killer": slot(p), "position": e.get("position")}
                    for p in players for e in p.get("events") or [] if e.get("type") == "CHAMPION_KILL"),
                   key=lambda k: k["t"])
    keys = {"towers": "towerKills", "inhibitors": "inhibitorKills", "dragons": "dragonKills", "elders": "elderDrakeKills",
            "barons": "baronKills", "heralds": "heraldKills", "atakhans": "atakhanKills"}
    teams = [t for t in game.get("teams") or [] if t.get("side") in ("blue", "red")]
    return {
        "detailed": bool(game.get("detailedStats")), "length_sec": game.get("length"),
        "objectives": objectives, "kills": kills,
        "positions": {slot(p): [[round(f["timestamp"] / 1000, 3), f["position"]["x"], f["position"]["y"]]
                                for f in p.get("frames") or [] if f.get("position")] for p in players},
        "gold": {t["side"]: [[round(f["timestamp"] / 1000, 3), f["gold"]] for f in t.get("frames") or []] for t in teams},
        "totals": {t["side"]: {k: t.get(v) for k, v in keys.items()} for t in teams},
    }
