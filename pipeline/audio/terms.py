"""받아쓰기 고유명사 교정 (코드, LLM 없음). asr 단계 끝에서 header 단어로 대사를 고친다.

whisper 는 챔피언·선수·팀 이름을 비슷한 소리로 잘못 적는다 (나필이 → 나피리, 런블 → 럼블, 하나 생명 → 한화생명).
대사의 단어(1~3단어 묶음)를 사전 단어와 한글 자모 단위로 비교해, 충분히 비슷하고 후보가 하나일 때만 바꾼다.

사전 (우선순위 순, 뒤로 갈수록 더 비슷해야 바꾼다)
1. 이 경기 챔피언 10개 (header players[].champion.ko)
2. extra 로 받은 이름 (api 가 넘기는 선수 한국어 이름표·팀 이름. 예: [{"text": "룰러", "kind": "player", "slot": "blue_bot"}])
3. 밴·피어리스로 못 쓰는 챔피언, 스킬 이름 (header)

결과 줄: text(고친 문장), raw(원문, 고친 게 있을 때만), fixes([{from, to, kind, slot, score}]).

    python -m pipeline.audio.terms outputs/<영상>/work     # 이미 있는 asr.json 에 다시 적용 (whisper 는 안 돌림)
"""

import json
import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

CHO = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
JUNG = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
JONG = " ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
PARTICLES = ("에게서", "한테서", "에게", "한테", "께서", "까지", "부터", "처럼", "보다", "으로", "이가", "이는", "이를",
             "이의", "이도", "이랑", "이고", "이에요", "이죠", "이", "가", "은", "는", "을", "를", "의", "와", "과", "도",
             "만", "로", "랑", "요")
TIER_BONUS = {1: 0.0, 2: 0.0, 3: 0.08}  # 사전 단계별로 더 비슷해야 바꾼다
DEFAULT_MIN_SCORE = 0.78


@dataclass
class Term:
    text: str         # 정식 표기
    kind: str         # champion | player | team | skill
    tier: int
    slot: str | None = None

    @property
    def key(self) -> str:
        return jamo(self.text.replace(" ", ""))


def jamo(s: str) -> str:
    """한글 → 자모 문자열 (럼블 → ㄹㅓㅁㅂㅡㄹ). 한글이 아닌 글자는 그대로."""
    out = []
    for ch in s:
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:
            out += [CHO[code // 588], JUNG[code % 588 // 28], JONG[code % 28].strip()]
        else:
            out.append(ch)
    return "".join(out)


def build_terms(header: dict | None, extra: list[dict] | None = None) -> list[Term]:
    """header(+ extra) → 사전. 같은 표기는 우선순위가 높은 쪽만 남긴다."""
    terms: list[Term] = []
    h = header or {}
    for p in h.get("players", []):
        champ = p.get("champion") or {}
        slot = f"{p['side']}_{p['role']}"
        if champ.get("ko"):
            terms.append(Term(champ["ko"], "champion", 1, slot))
        for skill in (champ.get("skills") or {}).values():
            if skill:
                terms.append(Term(skill, "skill", 3, slot))
    terms += _extra_terms(h, extra or [])
    for side in (h.get("bans") or {}).values():
        terms += [Term(b["ko"], "champion", 3) for b in side if b.get("ko")]
    terms += [Term(c["ko"], "champion", 3) for c in (h.get("series") or {}).get("fearless_unavailable", []) if c.get("ko")]
    seen, out = set(), []
    for t in sorted(terms, key=lambda t: t.tier):
        if t.text not in seen and len(t.text.replace(" ", "")) >= 2:
            seen.add(t.text)
            out.append(t)
    return out


def _extra_terms(h: dict, extra: list[dict]) -> list[Term]:
    """terms_extra [{text(한국어), kind(player|team), name?(영어), slot?}] → 사전 항목.

    name 이 있으면 이 경기 header 의 선수·팀 이름과 맞는 것만 쓰고 slot 을 붙인다 (api 는 전체 사전을 보내므로
    다른 경기 선수 이름으로 잘못 고치지 않게). name 이 없거나 header 에 선수 정보가 없으면 그대로 쓴다.
    """
    players = {(p.get("name") or "").lower(): f"{p['side']}_{p['role']}" for p in h.get("players", []) if p.get("name")}
    teams = {((t or {}).get("name") or "").lower(): side for side, t in (h.get("teams") or {}).items()}
    out = []
    for e in extra:
        name = (e.get("name") or "").lower()
        kind = e.get("kind", "player")
        known = players if kind == "player" else teams
        if name and known:
            if name not in known:
                continue
            out.append(Term(e["text"], kind, 2, known[name] if kind == "player" else None))
        else:
            out.append(Term(e["text"], kind, 2, e.get("slot")))
    return out


def split_particle(word: str) -> tuple[str, str]:
    """단어 → (본말, 조사). 본말이 2글자 이상 남을 때만 뗀다."""
    for p in PARTICLES:
        if word.endswith(p) and len(word) - len(p) >= 2:
            return word[: -len(p)], p
    return word, ""


# 고친 이름의 받침에 맞춰 조사를 바꾼다 (나필이는 → 나피리는, 런블을 → 럼블을)
AFTER_VOWEL = {"이는": "는", "이가": "가", "이를": "를", "이도": "도", "이랑": "랑", "이의": "의", "이고": "고",
               "이에요": "예요", "이죠": "죠", "이": "", "은": "는", "을": "를", "과": "와", "으로": "로"}
AFTER_CONSONANT = {"는": "은", "를": "을", "와": "과", "가": "이", "로": "으로", "랑": "이랑"}


def fit_particle(term: str, particle: str) -> str:
    last = ord(term[-1]) - 0xAC00 if term and "가" <= term[-1] <= "힣" else -1
    if last < 0 or not particle:
        return particle
    final = last % 28
    if final == 0:
        return AFTER_VOWEL.get(particle, particle)
    if particle == "로" and final == 8:  # ㄹ 받침은 '로'
        return particle
    return AFTER_CONSONANT.get(particle, particle)


def _syllables(s: str) -> int:
    return sum(1 for ch in s if "가" <= ch <= "힣")


def best_term(candidate: str, terms: list[Term], min_score: float) -> tuple[Term, float] | None:
    """candidate(띄어쓰기 없는 글자) 와 가장 비슷한 사전 단어. 후보가 둘 이상 비슷하면 None (애매하면 안 바꾼다)."""
    n = _syllables(candidate)
    if n < 2:
        return None
    key = jamo(candidate)
    scored = []
    for t in terms:
        tn = _syllables(t.text)
        if abs(tn - n) > 1:
            continue
        need = min_score + TIER_BONUS[t.tier] + (0.05 if min(n, tn) == 2 else 0.0)  # 두 글자는 더 엄격하게
        score = SequenceMatcher(None, key, t.key).ratio()
        if score >= need:
            scored.append((score - TIER_BONUS[t.tier], score, t))
    if not scored:
        return None
    scored.sort(key=lambda x: -x[0])
    if len(scored) > 1 and scored[0][0] - scored[1][0] < 0.03 and scored[0][2].text != scored[1][2].text:
        return None
    return scored[0][2], round(scored[0][1], 2)


def correct_text(text: str, terms: list[Term], min_score: float = DEFAULT_MIN_SCORE) -> tuple[str, list[dict]]:
    """대사 한 줄 → (고친 문장, fixes). 1~3단어 묶음을 긴 것부터 맞춰 본다. 이미 정확한 단어는 그대로."""
    words = text.split(" ")
    exact = {t.text.replace(" ", "") for t in terms}
    out, fixes, i = [], [], 0
    while i < len(words):
        done = False
        for size in (3, 2, 1):
            if i + size > len(words):
                continue
            chunk = words[i : i + size]
            if size > 1 and any(_syllables(w) < 2 or split_particle(w)[0] in exact for w in chunk):
                continue  # 여러 단어 묶음은 두 글자 이상 조각끼리만 (앞뒤 단어를 삼키지 않게)
            core, particle = split_particle(re.sub(r"[.,!?~…]+$", "", chunk[-1]))
            tail = chunk[-1][len(core) + len(particle):]
            cand = "".join(chunk[:-1]) + core
            if not re.fullmatch(r"[가-힣]+", cand or "-"):
                continue
            if cand in exact:
                if size == 1:
                    break
                continue
            hit = best_term(cand, terms, min_score)
            if hit and (size == 1 or hit[0].text.count(" ") + 1 >= size - 1):
                term, score = hit
                out.append(term.text + fit_particle(term.text, particle) + tail)
                fixes.append({"from": " ".join(chunk[:-1] + [core]), "to": term.text, "kind": term.kind,
                              "slot": term.slot, "score": score})
                i += size
                done = True
                break
        if not done:
            out.append(words[i])
            i += 1
    return " ".join(out), fixes


def correct_lines(lines: list[dict], terms: list[Term], min_score: float = DEFAULT_MIN_SCORE) -> list[dict]:
    """asr 대사 목록에 교정을 적용한다. raw 가 이미 있으면 원문(raw)에서 다시 고친다."""
    out = []
    for ln in lines:
        raw = ln.get("raw") or ln["text"]
        text, fixes = correct_text(raw, terms, min_score)
        new = {k: v for k, v in ln.items() if k not in ("raw", "fixes")}
        new["text"] = text
        if fixes:
            new["raw"], new["fixes"] = raw, fixes
        out.append(new)
    return out


def main() -> None:
    """python -m pipeline.audio.terms outputs/<영상>/work [terms_extra.json]
    terms_extra 는 파일로 주거나, 없으면 asr.json 을 만들 때 쓴 설정(params.terms_extra)을 그대로 쓴다."""
    work = Path(sys.argv[1])
    asr_path = work / "asr.json"
    data = json.loads(asr_path.read_text(encoding="utf-8"))
    header = json.loads((work / "header.json").read_text(encoding="utf-8")).get("result")
    if len(sys.argv) > 2:
        extra = json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))
        data.setdefault("params", {})["terms_extra"] = extra
    else:
        extra = (data.get("params") or {}).get("terms_extra") or []
    terms = build_terms(header, extra)
    data["result"]["lines"] = correct_lines(data["result"]["lines"], terms)
    asr_path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    for ln in data["result"]["lines"]:
        for f in ln.get("fixes", []):
            print(f"{ln['start_sec']:8.1f}  {f['from']} → {f['to']} ({f['kind']}, {f['score']})")


if __name__ == "__main__":
    main()
