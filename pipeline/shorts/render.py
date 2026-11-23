"""숏폼 렌더: 원본 영상의 구간들을 세로(9:16) 영상 하나로 만든다 (ffmpeg 한 번).

화면 (720x1280, 검은 배경, 11-23 참고 쇼츠 구성):
- 위: 제목 1~2줄 (title_lines. 두 줄이면 첫 줄 흰색 = 경기 정보, 둘째 줄 노란색 = 숏폼 제목)
- 가운데: 1920x1080 중계 화면 가운데 1180x1080 을 잘라 720x658 로 (점수판·KDA 표가 보이게), y=320
- 게임 화면 아래쪽에 설명 자막 (captions: 원본 영상 기준 [start_sec, end_sec] 동안 1~2줄)
글자는 나눔스퀘어 Bold + 검은 테두리. 구간이 여러 개면 순서대로 이어 붙이고, 자막 시각은 이어 붙인 영상 기준으로 옮긴다.
BGM(bgm.pick 으로 고른 곡)이 있으면 반복해 깔고, 해설(중계 소리)이 나오면 자동으로 줄인다 (sidechaincompress), 끝 1초는 줄여 끝냄.

결과는 임시 파일에 쓴 뒤 이름을 바꿔, 중간에 실패해도 반쯤 쓴 파일이 결과로 보이지 않게 한다.
"""

import json
import logging
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger(__name__)

W, H = 720, 1280
CROP = (370, 0, 1180, 1080)        # x, y, w, h (1920x1080 기준): 가운데, 원본 높이 전체
GAME_W, GAME_H, GAME_Y = 720, 658, 320
FONT = "/usr/share/fonts/truetype/nanum/NanumSquareB.ttf"
TITLE_STYLE = [(54, "white", 120), (58, "0xFFD400", 196)]   # (글자 크기, 색, y) 줄마다. 한 줄이면 둘째 것
CAPTION_STYLE = [(46, "white", 800), (46, "white", 860)]
BGM_VOLUME = 0.25        # 해설이 없을 때 BGM 크기 (원음 대비)


def render(video: Path, out_path: Path, segments: list[dict], crf: int = 23, preset: str = "veryfast",
           title_lines: list[str] | None = None, captions: list[dict] | None = None, bgm: dict | None = None) -> dict:
    """구간들을 세로 숏폼 mp4 하나로 렌더한다.

    Args:
        video: 원본 영상 (1920x1080).
        out_path: 결과 mp4 경로. 옆에 같은 이름의 .json(렌더 기록)도 쓴다.
        segments: [{start_sec, end_sec}] 원본 기준, 시간순·겹치지 않게.
        title_lines: 위 제목 0~2줄.
        captions: [{start_sec, end_sec, lines: [1~2줄]}] 원본 기준. 구간 밖 부분은 빠진다.
        bgm: 깔 곡 {file, title, source, license, attribution} (bgm.pick). 없으면 중계 소리만.

    Returns:
        렌더 기록 {duration_sec, render_sec, size_bytes, segments, bgm}.

    Raises:
        ValueError: 구간이 비었거나 순서가 틀림. (길이·개수 제한은 없다, 11-23)
        RuntimeError: ffmpeg 실패 (마지막 로그 몇 줄 포함).
    """
    segs = _check_segments(segments)
    total = sum(e - s for s, e in segs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    part = out_path.with_name(f".{out_path.stem}.part.mp4")  # 같은 폴더에 써야 이름 바꾸기가 한 번에 된다
    with tempfile.TemporaryDirectory() as tmp:
        texts = _text_layers(segs, title_lines or [], captions or [], Path(tmp))
        res = subprocess.run(_ffmpeg_cmd(video, segs, part, crf, preset, texts, (bgm or {}).get("file")),
                             capture_output=True, text=True)
    if res.returncode != 0:
        part.unlink(missing_ok=True)
        tail = "\n".join(res.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"ffmpeg 실패 (코드 {res.returncode}): {tail}")
    part.replace(out_path)
    meta = {"duration_sec": round(total, 2), "render_sec": round(time.monotonic() - t0, 2),
            "size_bytes": out_path.stat().st_size, "segments": [{"start_sec": s, "end_sec": e} for s, e in segs],
            "bgm": {k: v for k, v in bgm.items() if k != "file"} if bgm else None}
    out_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("숏폼 %s: %.1f초 영상, 렌더 %.1f초", out_path.name, total, meta["render_sec"])
    return meta


def _check_segments(segments: list[dict]) -> list[tuple[float, float]]:
    segs = [(float(s["start_sec"]), float(s["end_sec"])) for s in segments]
    if not segs:
        raise ValueError("구간이 없습니다")
    for (s, e), nxt in zip(segs, segs[1:] + [None]):
        if s < 0 or e <= s:
            raise ValueError(f"잘못된 구간: {s}~{e}")
        if nxt and nxt[0] < e:
            raise ValueError("구간은 시간순이고 겹치지 않아야 합니다")
    return segs


def output_spans(segs: list[tuple[float, float]], a: float, b: float) -> list[tuple[float, float]]:
    """원본 [a, b] → 이어 붙인 영상 안의 시각 구간들 (구간 밖 부분은 빠짐)."""
    out, offset = [], 0.0
    for s, e in segs:
        lo, hi = max(a, s), min(b, e)
        if hi > lo:
            out.append((round(lo - s + offset, 3), round(hi - s + offset, 3)))
        offset += e - s
    return out


def _text_layers(segs, title_lines: list[str], captions: list[dict], tmp: Path) -> list[dict]:
    """drawtext 층 목록 [{file, size, color, y, enable}]. 글자는 파일로 넘긴다 (따옴표·콜론 이스케이프를 피함)."""
    layers = []

    def add(text, style, enable=None):
        path = tmp / f"t{len(layers)}.txt"
        path.write_text(text, encoding="utf-8")
        size, color, y = style
        layers.append({"file": str(path), "size": size, "color": color, "y": y, "enable": enable})

    lines = [t.strip() for t in title_lines if t and t.strip()][:2]
    styles = TITLE_STYLE if len(lines) == 2 else TITLE_STYLE[1:]
    for text, style in zip(lines, styles):
        add(text, style)
    for c in captions:
        spans = output_spans(segs, float(c["start_sec"]), float(c["end_sec"]))
        if not spans:
            continue
        enable = "+".join(f"between(t,{a},{b})" for a, b in spans)
        for text, style in zip([t for t in c.get("lines", []) if t][:2], CAPTION_STYLE):
            add(text, style, enable)
    return layers


def _ffmpeg_cmd(video: Path, segs: list[tuple[float, float]], out: Path, crf: int, preset: str,
                texts: list[dict] | None = None, bgm_file: str | None = None) -> list[str]:
    """구간마다 입력을 하나씩 열고(-ss/-to 로 빠르게 찾기), 가운데를 잘라 이어 붙인 뒤 검은 9:16 화면에 얹고 글자를 쓴다."""
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for s, e in segs:
        cmd += ["-ss", f"{s:.3f}", "-to", f"{e:.3f}", "-i", str(video)]
    if bgm_file:
        cmd += ["-stream_loop", "-1", "-i", str(bgm_file)]
    cx, cy, cw, ch = CROP
    parts, concat_in = [], ""
    for i in range(len(segs)):
        parts.append(
            f"[{i}:v]scale=1920:1080,setsar=1,crop={cw}:{ch}:{cx}:{cy},scale={GAME_W}:{GAME_H},fps=30[v{i}];"
            f"[{i}:a]aresample=48000,aformat=channel_layouts=stereo[a{i}]"
        )
        concat_in += f"[v{i}][a{i}]"
    graph = ";".join(parts) + f";{concat_in}concat=n={len(segs)}:v=1:a=1[gv][ca]"
    graph += f";color=c=black:s={W}x{H}:r=30[bg];[bg][gv]overlay=0:{GAME_Y}:shortest=1"
    for t in texts or []:
        enable = f":enable='{t['enable']}'" if t["enable"] else ""
        graph += (f",drawtext=fontfile={FONT}:textfile='{t['file']}':fontsize={t['size']}:fontcolor={t['color']}"
                  f":borderw=4:bordercolor=black:x=(w-text_w)/2:y={t['y']}{enable}")
    graph += ",format=yuv420p[vout]"
    audio = "[ca]"
    if bgm_file:
        total = sum(e - s for s, e in segs)
        n = len(segs)
        graph += (f";[ca]asplit=2[voice][sc];[{n}:a]aresample=48000,aformat=channel_layouts=stereo,"
                  f"volume={BGM_VOLUME},atrim=0:{total:.3f},asetpts=N/SR/TB[bg];"
                  f"[bg][sc]sidechaincompress=threshold=0.03:ratio=10:attack=20:release=500[bgd];"
                  f"[voice][bgd]amix=inputs=2:duration=first:normalize=0,"
                  f"afade=t=out:st={max(0.0, total - 1):.3f}:d=1[aout]")
        audio = "[aout]"
    cmd += ["-filter_complex", graph, "-map", "[vout]", "-map", audio,
            "-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-c:a", "aac", "-b:a", "128k",
            "-movflags", "+faststart", str(out)]
    return cmd
