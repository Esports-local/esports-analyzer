"""hls 단계: 원본을 재인코딩해 전환 경계에서 자른 ts 조각과 m3u8 을 만든다.

ts 나누는 규칙
- 모든 전환 경계에서 자른다. 하드컷(cut)은 한 시점이라 경계만 생기고 ts 는 없다.
- 점진 전환(gradual) 구간은 별도 ts 로 만든다 (어느 segment 에도 속하지 않음).
- 한 구간이 max_ts_sec 보다 길면 균등 분할한다. 9.0초 → 3.0 / 3.0 / 3.0
- 경계는 프레임 시각에 맞춘다 (분석과 같은 fps 로 CFR 재인코딩).

인코딩
- 경계마다 키프레임(I-frame)을 강제로 넣고(-force_key_frames), 그 사이는 P/B 프레임만 쓴다
  (자동 장면 키프레임 끔: -sc_threshold 0). ts 하나 = I-frame 1개로 시작.
- 해상도는 원본 그대로. H.264 + AAC.

출력 (outputs/<영상명>/hls/)
- seg_00001.ts …
- index.m3u8   ts 목록. #EXTINF 제목 칸에 seg=<segment id> 또는 transition=gradual[ black]
- master.m3u8  RESOLUTION, FRAME-RATE, CODECS, BANDWIDTH
"""

import csv
import math
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from pipeline.io import load_stage
from pipeline.media import fps_of, probe, video_stream
from pipeline.schema import SegmentsResult

DEFAULT_PARAMS = {
    "max_ts_sec": 4.0,
    "crf": 20,
    "preset": "medium",
    "audio_bitrate": "128k",
}

# 강제 키프레임·분할 시각을 프레임 시각보다 조금 앞에 둬서, 부동소수 오차로 다음 프레임에 걸리지 않게 한다
_EPS = 1e-3


@dataclass
class Piece:
    """ts 하나가 될 구간."""

    start_sec: float
    end_sec: float
    label: str  # seg=3 / transition=gradual / transition=gradual black


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """hls 단계를 실행한다.

    Args:
        video: 원본 영상 경로.
        work_dir: 회차 작업 디렉터리. work/segments.json 이 있어야 한다.
        params: DEFAULT_PARAMS 를 덮어쓸 설정 (max_ts_sec, crf, preset, audio_bitrate).

    Returns:
        hls/index.m3u8 경로.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    info = probe(video)
    fps = fps_of(video_stream(info))
    has_audio = any(s["codec_type"] == "audio" for s in info["streams"])
    seg = load_stage(work_dir, "segments", SegmentsResult).result

    pieces = plan_pieces(seg, fps, p["max_ts_sec"])
    out_dir = work_dir / "hls"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("seg_*.ts"):
        old.unlink()

    actual = encode(video, out_dir, pieces, fps, has_audio, p)
    if len(actual) != len(pieces):
        raise RuntimeError(f"ts 개수가 계획과 다름: 계획 {len(pieces)}, 실제 {len(actual)}")

    index = out_dir / "index.m3u8"
    write_media_playlist(index, [(name, dur, piece.label) for (name, dur), piece in zip(actual, pieces)])
    write_master_playlist(out_dir / "master.m3u8", out_dir, index, fps, has_audio)
    return index


def plan_pieces(seg: SegmentsResult, fps: float, max_sec: float) -> list[Piece]:
    """segments 결과로 ts 구간 목록을 만든다.

    segment 와 점진 전환 구간을 시간순으로 늘어놓고, max_sec 보다 긴 구간은 균등 분할한다.
    모든 경계는 프레임 시각(k / fps)에 맞추고, 길이가 0 이 된 구간은 버린다.

    Args:
        seg: segments 단계 결과.
        fps: 프레임레이트 (재인코딩 fps 와 같아야 한다).
        max_sec: ts 최대 길이(초).

    Returns:
        시간순 Piece 목록. 서로 겹치지 않고 빈틈 없이 이어진다.
    """
    spans = [(s.start_sec, s.end_sec, f"seg={s.id}") for s in seg.segments]
    for t in seg.transitions:
        if t.kind == "gradual":
            spans.append((t.start_sec, t.end_sec, "transition=gradual" + (" black" if t.black else "")))
    spans.sort()

    def snap(x: float) -> float:
        return round(x * fps) / fps

    pieces: list[Piece] = []
    for start, end, label in spans:
        n = max(1, math.ceil((end - start) / max_sec - 1e-9))
        edges = [snap(start + (end - start) * k / n) for k in range(n + 1)]
        for a, b in zip(edges, edges[1:]):
            if b - a > 0.5 / fps:
                pieces.append(Piece(round(a, 6), round(b, 6), label))
    return pieces


def encode(video: Path, out_dir: Path, pieces: list[Piece], fps: float, has_audio: bool, p: dict) -> list[tuple[str, float]]:
    """재인코딩하면서 계획한 경계에서 ts 로 자른다.

    Args:
        video: 원본 영상 경로.
        out_dir: hls 출력 디렉터리.
        pieces: plan_pieces() 결과.
        fps: 출력 프레임레이트 (CFR).
        has_audio: 오디오 트랙 유무.
        p: 설정.

    Returns:
        (ts 파일 이름, 실제 길이(초)) 목록. 길이는 ts 의 프레임 수 ÷ fps (CFR 이라 정확하다).
        ffmpeg 분할 목록의 시각은 B-frame 지연만큼 밀려 있어 쓰지 않는다.
    """
    cuts = [pc.start_sec for pc in pieces[1:]]
    times = ",".join(f"{max(t - _EPS, 0):.6f}" for t in cuts)
    gop = max(int(fps * (p["max_ts_sec"] + 1)), 1)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", "0:v:0"]
    if has_audio:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", p["audio_bitrate"], "-ac", "2", "-ar", "48000"]
    cmd += [
        "-vf", f"fps={fps!r}", "-c:v", "libx264", "-preset", p["preset"], "-crf", str(p["crf"]),
        "-pix_fmt", "yuv420p", "-sc_threshold", "0", "-g", str(gop), "-keyint_min", "1",
        "-avoid_negative_ts", "make_zero",
    ]
    if times:
        cmd += ["-force_key_frames", times]
    list_csv = out_dir / "segments.csv"
    cmd += [
        "-f", "segment", "-segment_format", "mpegts",
        "-segment_list", str(list_csv), "-segment_list_type", "csv",
        "-reset_timestamps", "0",
    ]
    if times:
        cmd += ["-segment_times", times]
    cmd += [str(out_dir / "seg_%05d.ts")]
    subprocess.run(cmd, check=True)

    with list_csv.open() as f:
        names = [r[0] for r in csv.reader(f) if r]
    list_csv.unlink()
    # ffmpeg 는 0 부터 번호를 매긴다 → 1 부터로 바꾼다 (겹치지 않게 임시 이름을 거친다)
    for i, name in enumerate(names, 1):
        (out_dir / name).rename(out_dir / f"seg_{i:05d}.ts.tmp")
    result = []
    for i in range(1, len(names) + 1):
        final = out_dir / f"seg_{i:05d}.ts"
        (out_dir / f"seg_{i:05d}.ts.tmp").rename(final)
        result.append((final.name, count_frames(final) / fps))
    return result


def count_frames(ts: Path) -> int:
    """ts 파일의 영상 프레임 수를 센다 (패킷 수, 디코딩 없음).

    Args:
        ts: ts 파일 경로.

    Returns:
        영상 프레임 수.
    """
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets",
                          "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(ts)],
                         check=True, capture_output=True, text=True).stdout
    # mpegts 는 program 과 stream 으로 같은 값이 두 번 나온다 → 첫 값만 쓴다
    return int(next(line for line in out.splitlines() if line.strip()).split(",")[0])


def write_media_playlist(path: Path, entries: list[tuple[str, float, str]]) -> None:
    """ts 목록 m3u8(index.m3u8) 을 쓴다.

    Args:
        path: 출력 경로.
        entries: (ts 파일 이름, 길이(초), 표시) 목록. 표시는 #EXTINF 제목 칸에 들어간다 (재생에는 영향 없음).
    """
    target = math.ceil(max(d for _, d, _ in entries)) if entries else 1
    lines = ["#EXTM3U", "#EXT-X-VERSION:3", f"#EXT-X-TARGETDURATION:{target}",
             "#EXT-X-MEDIA-SEQUENCE:0", "#EXT-X-PLAYLIST-TYPE:VOD"]
    for name, dur, label in entries:
        lines += [f"#EXTINF:{dur:.6f},{label}", name]
    lines.append("#EXT-X-ENDLIST")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_master_playlist(path: Path, out_dir: Path, index: Path, fps: float, has_audio: bool) -> None:
    """master.m3u8 을 쓴다 (해상도·프레임레이트·코덱·대역폭).

    Args:
        path: 출력 경로.
        out_dir: ts 가 있는 디렉터리.
        index: index.m3u8 경로 (master 에서 상대 경로로 가리킴).
        fps: 프레임레이트.
        has_audio: 오디오 포함 여부 (CODECS 에 mp4a 추가).
    """
    entries = read_playlist(index)
    first = out_dir / entries[0]["file"]
    v = video_stream(probe(first))
    codecs = [avc1_codec(v)] + (["mp4a.40.2"] if has_audio else [])
    peak = max((out_dir / e["file"]).stat().st_size * 8 / max(e["duration_sec"], 1e-3) for e in entries)
    total = sum((out_dir / e["file"]).stat().st_size for e in entries) * 8
    avg = total / max(sum(e["duration_sec"] for e in entries), 1e-3)
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:3",
        f'#EXT-X-STREAM-INF:BANDWIDTH={int(peak)},AVERAGE-BANDWIDTH={int(avg)},'
        f'RESOLUTION={v["width"]}x{v["height"]},FRAME-RATE={fps:.3f},CODECS="{",".join(codecs)}"',
        index.name,
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def avc1_codec(stream: dict) -> str:
    """H.264 스트림 정보로 HLS CODECS 문자열(avc1.PPCCLL)을 만든다.

    Args:
        stream: ffprobe 영상 스트림 정보 (profile, level).

    Returns:
        예: High, level 4.0 → "avc1.640028".
    """
    profile_idc = {"Baseline": 0x42, "Constrained Baseline": 0x42, "Main": 0x4D, "High": 0x64}.get(
        stream.get("profile", ""), 0x64)
    constraint = 0x40 if stream.get("profile") == "Constrained Baseline" else 0x00
    level = int(stream.get("level", 40))
    return f"avc1.{profile_idc:02x}{constraint:02x}{level:02x}"


def read_playlist(index: Path) -> list[dict]:
    """index.m3u8 을 읽어 ts 목록을 돌려준다. 다른 단계가 ts·segment 시각을 알아낼 때 쓴다.

    Args:
        index: index.m3u8 경로.

    Returns:
        ts 마다 {"id", "file", "start_sec", "end_sec", "duration_sec", "segment_id", "transition", "black"}.
        segment_id 는 seg=N 이면 N, 전환 ts 면 None. transition 은 전환 ts 면 "gradual", 아니면 None.
    """
    entries, t, dur, label = [], 0.0, None, ""
    for line in index.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#EXTINF:"):
            d, _, label = line[len("#EXTINF:"):].partition(",")
            dur = float(d)
        elif line and not line.startswith("#") and dur is not None:
            m = re.match(r"seg=(\d+)", label)
            tr = re.match(r"transition=(\w+)", label)
            entries.append({
                "id": len(entries) + 1, "file": line,
                "start_sec": round(t, 6), "end_sec": round(t + dur, 6), "duration_sec": dur,
                "segment_id": int(m.group(1)) if m else None,
                "transition": tr.group(1) if tr else None,
                "black": "black" in label.split(),
            })
            t += dur
            dur = None
    return entries
