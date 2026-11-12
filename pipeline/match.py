"""경기 정보: 날짜·세트·두 팀을 받아 진영·포지션별 선수·챔피언·승패를 가져와 match.json 으로 저장한다.

    python -m pipeline.match 261108 2 T1 BLG            # → data/matches/<경기 id>.json
    python -m pipeline.match 261108 2 T1 BLG --refresh  # 저장본 무시하고 다시 받기

출처
- Leaguepedia Cargo API (lol.fandom.com): ScoreboardGames(세트), ScoreboardPlayers(선수), Teams(약칭).
  익명 요청은 횟수 제한이 빡빡하므로 요청 사이에 간격을 두고, 한 번 받은 경기는 파일을 다시 쓴다.
  경기 후 사람이 채우는 곳이라 막 끝난 경기는 아직 없을 수 있다.
- Riot Data Dragon: 챔피언 영어 ↔ 한국어 이름.
- 선수 닉네임의 한국어 표기(페이커 등)는 자동으로 가져올 곳이 없어 name_ko 를 비워 두고 직접 채운다.

match.json
{
  "schema_version": 1, "source": "leaguepedia", "game_id": "...", "match_id": "...",
  "tournament": "...", "date_utc": "2026-11-08 09:12:00", "game": 2, "winner": "blue",
  "blue": {"team": "T1", "team_full": "T1",
           "players": {"top": {"name": "Doran", "name_ko": null, "champion": "Olaf", "champion_ko": "올라프",
                               "kda": "1/2/0"}, ...}},
  "red":  {...}
}
날짜는 한국 시간 기준 yymmdd. 세트는 그 경기(매치) 안의 몇 번째 게임인지.
"""

import argparse
import json
import logging
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx

log = logging.getLogger(__name__)

API = "https://lol.fandom.com/api.php"
DDRAGON = "https://ddragon.leagueoflegends.com"
USER_AGENT = "esports-analyzer/0.1 (research; match metadata)"
KST = timezone(timedelta(hours=9))
ROLES = {"Top": "top", "Jungle": "jungle", "Mid": "mid", "Bot": "bot", "Support": "support"}
SCHEMA_VERSION = 1


def fetch_match(date: str, game: int, team_a: str, team_b: str,
                out_dir: Path = Path("data/matches"), refresh: bool = False) -> Path:
    """경기 정보를 받아 저장한다. 같은 경기 파일이 있으면 요청하지 않고 그 경로를 돌려준다.

    Args:
        date: 한국 시간 기준 yymmdd (예: 261108).
        game: 매치 안의 세트 번호 (1부터).
        team_a, team_b: 팀 약칭이나 정식 이름 (순서·진영 무관, 대소문자 무관).
        out_dir: 저장 폴더.
        refresh: True 면 저장본이 있어도 다시 받는다.

    Returns:
        match.json 경로.

    Raises:
        LookupError: 조건에 맞는 경기를 찾지 못함.
        RuntimeError: API 오류 (요청 제한이 계속되는 경우 포함).
    """
    day = datetime.strptime(date, "%y%m%d").replace(tzinfo=KST)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = sorted(out_dir.glob(f"{day:%y%m%d}_*_g{game}.json"))
    for path in cache:
        m = json.loads(path.read_text(encoding="utf-8"))
        if not refresh and _same_teams(m, team_a, team_b):
            log.info("match: 저장본 사용 %s", path)
            return path

    api = Leaguepedia()
    # KST 하루는 UTC 로 전날 15시 ~ 당일 15시. 앞뒤 여유를 두고 찾은 뒤 한국 날짜로 거른다
    start, end = day - timedelta(days=1), day + timedelta(days=2)
    games = api.query("ScoreboardGames=SG",
                      "SG.GameId=GameId,SG.MatchId=MatchId,SG.Tournament=Tournament,SG.Team1=Team1,SG.Team2=Team2,"
                      "SG.WinTeam=WinTeam,SG.DateTime_UTC=DateTimeUTC,SG.N_GameInMatch=N",
                      f"SG.DateTime_UTC >= '{start:%Y-%m-%d}' AND SG.DateTime_UTC < '{end:%Y-%m-%d}'"
                      f" AND SG.N_GameInMatch = {int(game)}")
    games = [g for g in games if _kst_date(g["DateTimeUTC"]) == day.date()]
    if not games:
        raise LookupError(f"{date} {game}세트 경기가 Leaguepedia 에 없습니다 (아직 안 올라왔거나 날짜가 다름)")

    names = sorted({g[k] for g in games for k in ("Team1", "Team2")})
    shorts = {t["Name"]: t["Short"] for t in api.query(
        "Teams", "Name,Short", "Name IN (" + ",".join(_quote(n) for n in names) + ")")}
    found = [g for g in games if _matches(team_a, g, shorts) and _matches(team_b, g, shorts)]
    if not found:
        listed = ", ".join(f"{shorts.get(g['Team1'], g['Team1'])} vs {shorts.get(g['Team2'], g['Team2'])}" for g in games)
        raise LookupError(f"{date} {game}세트에 {team_a} vs {team_b} 경기가 없습니다. 그날 경기: {listed}")
    g = found[0]

    players = api.query("ScoreboardPlayers=SP",
                        "SP.Link=Link,SP.Champion=Champion,SP.Role=Role,SP.Side=Side,SP.Team=Team,"
                        "SP.Kills=Kills,SP.Deaths=Deaths,SP.Assists=Assists",
                        f"SP.GameId = {_quote(g['GameId'])}")
    champions_ko = champion_names_ko()

    match = {"schema_version": SCHEMA_VERSION, "source": "leaguepedia", "game_id": g["GameId"],
             "match_id": g["MatchId"], "tournament": g["Tournament"], "date_utc": g["DateTimeUTC"],
             "game": int(g["N"]), "winner": "blue" if g["WinTeam"] == g["Team1"] else "red"}
    for side, key, team in (("1", "blue", g["Team1"]), ("2", "red", g["Team2"])):
        rows = {}
        for p in players:
            if p["Side"] != side or p["Role"] not in ROLES:
                continue
            rows[ROLES[p["Role"]]] = {"name": _display(p["Link"]), "name_ko": None, "champion": p["Champion"],
                                      "champion_ko": champions_ko.get(_norm(p["Champion"])),
                                      "kda": f"{p['Kills']}/{p['Deaths']}/{p['Assists']}"}
        match[key] = {"team": shorts.get(team, team), "team_full": team,
                      "players": {r: rows[r] for r in ROLES.values() if r in rows}}

    path = out_dir / f"{day:%y%m%d}_{_slug(match['blue']['team'])}_{_slug(match['red']['team'])}_g{game}.json"
    path.write_text(json.dumps(match, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log.info("match: %s → %s", g["GameId"], path)
    return path


def match_lexicons(path: Path) -> dict[str, dict[str, str]]:
    """match.json 으로 roi_ocr 사전(players, teams)을 만든다. 형식은 roi_ocr.load_lexicons 결과와 같다."""
    m = json.loads(Path(path).read_text(encoding="utf-8"))
    players, teams = {}, {}
    for side in ("blue", "red"):
        teams[m[side]["team"].lower()] = m[side]["team"]
        teams[m[side]["team_full"].lower()] = m[side]["team"]
        for p in m[side]["players"].values():
            for key in (p["name"], p.get("name_ko")):
                if key:
                    players[key.lower()] = p["name"]
    return {"players": players, "teams": teams}


class Leaguepedia:
    """Cargo API 요청. 요청 사이 간격을 두고, 요청 제한에 걸리면 늘려 가며 다시 시도한다."""

    def __init__(self, interval: float = 3.0, retries: int = 5):
        self.client = httpx.Client(headers={"User-Agent": USER_AGENT}, timeout=30)
        self.interval = interval
        self.retries = retries
        self._last = 0.0

    def query(self, tables: str, fields: str, where: str, limit: int = 100) -> list[dict]:
        """cargoquery 결과 행 목록. 값은 모두 문자열."""
        params = {"action": "cargoquery", "format": "json", "tables": tables, "fields": fields,
                  "where": where, "limit": limit}
        wait = self.interval
        for attempt in range(self.retries):
            time.sleep(max(0.0, self._last + self.interval - time.monotonic()))
            self._last = time.monotonic()
            data = self.client.get(API, params=params).raise_for_status().json()
            err = data.get("error")
            if not err:
                return [row["title"] for row in data.get("cargoquery", [])]
            if err.get("code") != "ratelimited":
                raise RuntimeError(f"Leaguepedia 오류: {err.get('code')}: {err.get('info')}")
            wait *= 2
            log.warning("match: 요청 제한 → %.0f초 후 다시 (%d/%d)", wait, attempt + 1, self.retries)
            time.sleep(wait)
        raise RuntimeError("Leaguepedia 요청 제한이 계속됩니다. 잠시 뒤 다시 실행하세요")


def champion_names_ko() -> dict[str, str]:
    """Data Dragon 최신 버전에서 {정규화한 영어 이름: 한국어 이름}."""
    with httpx.Client(timeout=30) as c:
        version = c.get(f"{DDRAGON}/api/versions.json").raise_for_status().json()[0]
        en = c.get(f"{DDRAGON}/cdn/{version}/data/en_US/champion.json").raise_for_status().json()["data"]
        ko = c.get(f"{DDRAGON}/cdn/{version}/data/ko_KR/champion.json").raise_for_status().json()["data"]
    out = {}
    for cid, v in en.items():
        if cid in ko:
            out[_norm(v["name"])] = ko[cid]["name"]
            out[_norm(cid)] = ko[cid]["name"]
    return out


def _kst_date(utc: str):
    return datetime.strptime(utc, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc).astimezone(KST).date()


def _norm(s: str) -> str:
    return re.sub(r"[^0-9a-z]", "", s.lower())


def _aliases(team: str, shorts: dict[str, str]) -> set[str]:
    return {_norm(team), _norm(shorts.get(team, ""))} - {""}


def _matches(name: str, g: dict, shorts: dict[str, str]) -> bool:
    return _norm(name) in _aliases(g["Team1"], shorts) | _aliases(g["Team2"], shorts)


def _same_teams(m: dict, a: str, b: str) -> bool:
    have = {_norm(m[s][k]) for s in ("blue", "red") for k in ("team", "team_full")}
    return {_norm(a), _norm(b)} <= have


def _display(link: str) -> str:
    """'Bin (Chen Ze-Bin)' 같은 동명이인 구분 괄호를 뗀다."""
    return re.sub(r"\s*\(.*\)$", "", link).strip()


def _quote(s: str) -> str:
    return "'" + s.replace("\\", "\\\\").replace("'", "\\'") + "'"


def _slug(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "", s) or "team"


def main():
    parser = argparse.ArgumentParser(description="경기 정보 받기 (Leaguepedia + Data Dragon)")
    parser.add_argument("date", help="한국 시간 기준 yymmdd")
    parser.add_argument("game", type=int, help="세트 번호")
    parser.add_argument("team_a")
    parser.add_argument("team_b")
    parser.add_argument("--out-dir", type=Path, default=Path("data/matches"))
    parser.add_argument("--refresh", action="store_true", help="저장본이 있어도 다시 받기")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        path = fetch_match(args.date, args.game, args.team_a, args.team_b, args.out_dir, args.refresh)
    except LookupError as e:
        parser.exit(1, f"{e}\n")
    print(path)


if __name__ == "__main__":
    main()
