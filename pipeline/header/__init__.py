"""header 단계: 세트 영상의 경기 기본 정보 (숏폼·자막 교정 때 그 경기에서 기본으로 알아야 하는 것).

사람·LLM 입력 없이 op.gg(경기 페이지 + 세트 정보)와 Riot Data Dragon(챔피언 한국어 이름·스킬 이름)으로만 만든다.
영상 파일은 읽지 않는다. 설정으로 op.gg 경기 번호(match)와 세트(set)를 받는다 (api 가 영상 등록 때 넘김).

    python -m pipeline <영상> --only header --set header.match=32706 --set header.set=2

피어리스 드래프트(하드): 앞 세트에서 양 팀이 고른 챔피언은 이 세트에서 아무도 고를 수 없다. 밴은 넘어가지 않는다.
FEARLESS_LEAGUES 의 FEARLESS_SINCE 년 이후 경기에만 적용한다 (그 밖에는 series.fearless=false, 목록은 빈 칸).

op.gg·Data Dragon 요청이 실패해도 분석 작업을 멈추지 않는다: 결과에 error 를 남기고 나머지는 비운다.

결과 (work/header.json 의 result)
    match:   op.gg 번호·이름·대회·스테이지·연도·형식(BO5)·세트·패치·시작 시각·시리즈 최종 스코어·op.gg 주소·이 세트 영상 주소
    teams:   {blue|red: {name(약칭), full, opgg_id}}
    players: [{side, team, role, name, champion: {name, ko, skills: {P,Q,W,E,R}}}]  블루 먼저, 탑→서폿
    bans:    {blue|red: [{name, ko}]}
    result:  {winner: blue|red|None, length_sec}  이 세트 결과
    series:  {score_before, previous: [{set, blue, red, winner, picks, bans}], fearless, fearless_unavailable}
    source:  {opgg, ddragon_version}
"""

import logging
from pathlib import Path

import httpx

from pipeline.header import ddragon, opgg
from pipeline.io import save_stage
from pipeline.schema import HeaderResult

log = logging.getLogger(__name__)

DEFAULT_PARAMS = {"match": None, "set": None}
ROLE_ORDER = ["top", "jungle", "mid", "bot", "support"]
FEARLESS_LEAGUES = {"LCK", "LPL", "MSI", "Worlds"}  # op.gg league shortName
FEARLESS_SINCE = 2025
ERRORS = (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError)


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """work/header.json 을 만든다. 영상은 쓰지 않는다.

    Raises:
        ValueError: match·set 설정이 없음.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not p["match"] or not p["set"]:
        raise ValueError("header 단계는 match(op.gg 경기 번호)와 set(세트 번호) 설정이 필요합니다")
    try:
        result = build(p["match"], int(p["set"]))
        log.info("[header] %s %s세트: 선수 %d명, 앞 세트 %d개", result.match.get("name"), p["set"],
                 len(result.players), len(result.series.get("previous", [])))
    except ERRORS as e:
        log.warning("[header] op.gg·Data Dragon 에서 가져오지 못했습니다: %r", e)
        result = HeaderResult(error=f"{type(e).__name__}: {e}")
    return save_stage(work_dir, "header", result, p)


def build(match_ref: str | int, set_number: int) -> HeaderResult:
    """op.gg·Data Dragon 에서 받아 헤더를 만든다. Raises: ERRORS 중 하나."""
    url, match = opgg.fetch_match(match_ref)
    games = {}
    for n in range(1, set_number + 1):  # 이 세트와 앞 세트들 (피어리스·스코어)
        g = opgg.fetch_game(match["id"], n)
        if g:
            games[n] = opgg.parse_game(g)
    if set_number not in games:
        raise ValueError(f"op.gg 에 {set_number}세트 정보가 없습니다")
    version = ddragon.version_for(match.get("matchVersion") or "")
    champs = ddragon.champions(version)
    return assemble(url, match, games, set_number, champs, lambda cid: ddragon.skills(version, cid), version)


def is_fearless(league: str, year: int | None) -> bool:
    return league in FEARLESS_LEAGUES and (year or 0) >= FEARLESS_SINCE


def assemble(url: str, match: dict, games: dict[int, dict], set_number: int, champs: dict[str, dict],
             skills_of, version: str) -> HeaderResult:
    """받은 데이터로 헤더를 조립한다 (네트워크 없음, 테스트용으로 분리). skills_of(Data Dragon id) → 스킬 이름."""
    teams_by_id = {int(t["id"]): t for t in (match["homeTeam"], match["awayTeam"])}
    acr = lambda tid: (teams_by_id.get(tid) or {}).get("acronym")  # noqa: E731

    def champ(cid: str) -> dict:
        c = champs.get(cid) or {}
        return {"name": c.get("name", ""), "ko": c.get("ko", "")}

    def picks(g: dict) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {"blue": [], "red": []}
        for p in sorted(g["players"], key=lambda p: ROLE_ORDER.index(p["role"])):
            if p["champion_id"] in champs:
                out[p["side"]].append(champ(p["champion_id"]))
        return out

    def side_of(g: dict, tid: int | None) -> str | None:
        return next((s for s, t in g["sides"].items() if t == tid), None)

    serie = (match.get("tournament") or {}).get("serie") or {}
    league, year = (serie.get("league") or {}).get("shortName") or "", serie.get("year")
    fearless = is_fearless(league, year)
    g = games[set_number]
    score = {t["acronym"]: 0 for t in teams_by_id.values()}
    previous, unavailable = [], []
    for n in sorted(k for k in games if k < set_number):
        prev = games[n]
        if acr(prev["winner"]):
            score[acr(prev["winner"])] += 1
        sides = {s: acr(t) for s, t in prev["sides"].items()}
        pk = picks(prev)
        previous.append({"set": n, "blue": sides.get("blue"), "red": sides.get("red"),
                         "winner": acr(prev["winner"]), "picks": pk,
                         "bans": {s: [champ(c) for c in ids if c in champs] for s, ids in prev["bans"].items()}})
        if fearless:
            unavailable += [{**c, "set": n, "team": sides.get(side)} for side, cs in pk.items() for c in cs]

    def player(p: dict) -> dict:
        c = champs.get(p["champion_id"]) or {}
        return {"side": p["side"], "team": acr(g["sides"].get(p["side"])), "role": p["role"], "name": p["name"],
                "champion": {**champ(p["champion_id"]), "skills": skills_of(c["id"]) if c else {}}}

    return HeaderResult(
        match={"opgg_id": int(match["id"]), "name": match.get("name") or "", "league": league,
               "stage": (match.get("tournament") or {}).get("name") or "", "year": year,
               "format": f"BO{match.get('numberOfGames') or 1}", "set": set_number,
               "patch": match.get("matchVersion") or "", "begin_at": match.get("beginAt") or match.get("scheduledAt"),
               "score": {match["homeTeam"]["acronym"]: match.get("homeScore") or 0,
                         match["awayTeam"]["acronym"]: match.get("awayScore") or 0},
               "source_url": url, "video_url": opgg.set_videos(match).get(set_number)},
        teams={s: {"name": acr(t), "full": (teams_by_id.get(t) or {}).get("name"), "opgg_id": t}
               for s, t in g["sides"].items()},
        players=[player(p) for p in sorted(g["players"], key=lambda p: (p["side"] != "blue",
                                                                         ROLE_ORDER.index(p["role"])))],
        bans={s: [champ(c) for c in ids if c in champs] for s, ids in g["bans"].items()},
        result={"winner": side_of(g, g["winner"]), "length_sec": g["length_sec"]},
        series={"score_before": score, "previous": previous, "fearless": fearless,
                "fearless_unavailable": unavailable},
        source={"opgg": url, "ddragon_version": version},
    )
