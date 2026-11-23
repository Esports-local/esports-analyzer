"""모음형 숏폼 요청 → 조건. "룰러 킬 모음" → {killer: blue_bot, type: kill}.

LLM 은 요청 글과 이 경기 선수 목록만 보고 조건 JSON 을 만든다 (경기 데이터는 안 봄, 짧은 호출).
이벤트는 api 가 이 조건으로 고른다. 조건 선택칸이 기준이고, 이것은 선택칸을 채워 주는 용도다.
받은 값은 코드가 검사한다: 선수는 목록의 슬롯만, 종류·팀·태그는 정해진 값만, 제목은 10자 이내만 남긴다.
"""

from pipeline import llm

TYPES = ("kill", "objective", "tower")
TEAMS = ("blue", "red")
TAGS = ("first_blood", "multikill", "teamfight")
TITLE_MAX = 10

SYSTEM_PROMPT = """너는 LoL 중계 숏폼 요청을 조건으로 바꾼다. 입력은 요청 글과 이 경기 선수 목록(슬롯: "팀 이름 (챔피언)")이다.
조건 칸 (해당 없으면 빈 문자열):
- killer: 킬한 선수의 슬롯 (blue_top · blue_jungle · blue_mid · blue_bot · blue_support · red_… 중 목록에 있는 것).
  선수 이름, 챔피언 이름, 한국어 발음(룰러 → Ruler, 카나비 → Kanavi), 포지션(블루 원딜) 어느 것으로 말해도 된다.
- type: kill · objective · tower
- team: blue · red (팀 이름으로 말하면 목록에서 그 팀의 진영)
- tag: first_blood(퍼스트 블러드) · multikill(연속 킬, 더블킬 이상) · teamfight(한타)
- title: 이 모음 숏폼 제목, 한국어 공백 포함 10자 이내 (예: "룰러 킬 모음")
- note: 조건으로 못 나타내는 요청이면 무엇을 못 하는지 한 줄 (예: "데스·도움 기준은 아직 고를 수 없음"). 없으면 빈 문자열.
목록에 없는 선수를 지어내지 않는다."""

RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {k: {"type": "STRING"} for k in ("killer", "type", "team", "tag", "title", "note")},
    "required": ["killer", "type", "team", "tag", "title", "note"],
}


def parse(text: str, players: dict[str, str], client=None, model: str = llm.DEFAULT_MODEL) -> dict:
    """{filter: {killer, type, team, tag}, title, note, error}. LLM 을 못 쓰면 error 만."""
    out = {"filter": {}, "title": None, "note": None, "error": None}
    if client is None and not llm.available():
        out["error"] = "GEMINI_API_KEY 가 없어 요청을 해석할 수 없습니다"
        return out
    user = "요청: " + text.strip() + "\n선수 목록:\n" + "\n".join(f"{k}: {v}" for k, v in players.items())
    try:
        raw, _ = (client or llm.Gemini(model)).generate_json(SYSTEM_PROMPT, user, RESPONSE_SCHEMA, temperature=0.0)
    except Exception as e:
        out["error"] = f"LLM 호출 실패: {e}"
        return out
    d = llm.parse_json(raw)
    if d is None:
        out["error"] = "응답을 읽지 못함"
        return out
    return {**out, **clean(d, players)}


def clean(d: dict, players: dict[str, str]) -> dict:
    """LLM 조건 → 허용된 값만."""
    f = {}
    if d.get("killer") in players:
        f["killer"] = d["killer"]
    for key, allowed in (("type", TYPES), ("team", TEAMS), ("tag", TAGS)):
        if d.get(key) in allowed:
            f[key] = d[key]
    title = (d.get("title") or "").strip()
    return {"filter": f, "title": title if 0 < len(title) <= TITLE_MAX else None, "note": (d.get("note") or "").strip() or None}
