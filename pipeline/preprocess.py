"""preprocess 단계: 영상 기본 정보와 asr 입력 오디오를 준비한다.

1. ffprobe 로 길이·해상도·fps·코덱을 읽고, 패킷 시각 간격으로 VFR 여부를 판단한다.
   VFR 이어도 원본은 바꾸지 않는다. 프레임을 읽는 단계(segments, faces)가 fps 필터로 CFR 에 맞춰 읽는다.
2. 오디오를 16kHz 모노 wav(work/audio.wav)로 뽑는다 (Silero VAD, faster-whisper 입력). 음량 정규화는 하지 않는다.
"""

import subprocess
from pathlib import Path

from pipeline.io import save_stage
from pipeline.media import fps_of, is_vfr, probe, video_stream
from pipeline.schema import AudioFile, AudioInfo, PreprocessResult, VideoInfo

DEFAULT_PARAMS = {"audio_sample_rate": 16000, "audio_channels": 1}


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    p = {**DEFAULT_PARAMS, **(params or {})}
    info = probe(video)
    vs = video_stream(info)
    fps = fps_of(vs)
    a_stream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)

    audio, audio_file = None, None
    if a_stream is not None:
        audio = AudioInfo(codec=a_stream["codec_name"], sample_rate=int(a_stream["sample_rate"]),
                          channels=int(a_stream["channels"]))
        wav = work_dir / "work" / "audio.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(video), "-vn", "-map", "0:a:0",
                        "-ac", str(p["audio_channels"]), "-ar", str(p["audio_sample_rate"]),
                        "-c:a", "pcm_s16le", str(wav)], check=True)
        audio_file = AudioFile(path=str(wav.relative_to(work_dir)), sample_rate=p["audio_sample_rate"],
                               channels=p["audio_channels"])

    result = PreprocessResult(
        source_path=str(video),
        duration_sec=round(float(info["format"]["duration"]), 3),
        video=VideoInfo(codec=vs["codec_name"], width=int(vs["width"]), height=int(vs["height"]),
                        fps=round(fps, 3), vfr=is_vfr(video, fps)),
        audio=audio,
        audio_file=audio_file,
    )
    return save_stage(work_dir, "preprocess", result, p)
