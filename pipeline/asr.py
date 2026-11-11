"""asr 단계: 음성을 받아 적고, 화자 라벨을 붙여 대사 줄로 묶는다.

1. 받아쓰기: faster-whisper. 한국어 외 대사(일본어·영어 등)가 대사 단위로 섞이므로
   Silero VAD 로 말소리 덩어리를 나누고, 덩어리마다 언어를 감지한 뒤 그 언어로 받아 적는다 (단어 단위 시각 포함).
   Whisper 의 기본 동작(처음 30초로 언어 하나를 정함)이나 multilingual(30초 창마다 감지)은
   대사마다 언어가 바뀌면 다른 언어 대사를 빠뜨리거나 엉뚱한 언어로 적는다.
2. 화자 구분(선택): diarization 백엔드가 있으면 단어마다 가장 많이 겹치는 화자를 붙인다.
   - "none": 화자 구분 안 함 (speaker = None). 기본값
   - "pyannote": pyannote Community-1. 환경변수 HF_TOKEN 필요 (아직 검증 안 됨)
3. 대사 줄 만들기
   - 화자가 있으면: 같은 화자의 단어가 merge_gap_sec 이내로 이어지면 한 줄 (한 사람이 이어서 말한 덩어리).
   - 화자가 없으면: Whisper 가 나눈 구간을 그대로 한 줄로 쓴다 (서로 다른 사람의 대사가 합쳐지지 않게).
   - 컷·ts 경계는 고려하지 않는다 (대사를 자르지 않음).
4. 언어: 말소리 덩어리의 감지 결과를 대사 줄에 남긴다 (작업 파일 work/asr.json 에만 남음).

입력: work/audio.wav (preprocess)
출력: work/asr.json
"""

import os
import wave
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline.io import load_stage, save_stage
from pipeline.schema import AsrResult, Line, PreprocessResult, Word

DEFAULT_PARAMS = {
    "model": "small",  # faster-whisper 모델 크기 (small / medium / large-v3)
    "device": "auto",  # auto / cpu / cuda
    "compute_type": "default",  # default 면 cuda=float16, cpu=int8
    "beam_size": 5,
    "languages": ["ko", "ja", "en"],  # 감지 후보 언어. 이 중 확률이 가장 높은 것을 고른다
    "default_language": "ko",  # 감지가 불확실하거나 덩어리가 짧을 때 쓸 언어
    "language_min_prob": 0.7,  # default_language 가 아닌 언어로 바꾸려면 확률이 이 이상이어야 한다
    "min_detect_sec": 1.0,  # 이보다 짧은 덩어리는 감지하지 않고 default_language
    "vad_min_silence_ms": 500,  # 이 이상 조용하면 말소리 덩어리를 나눈다
    "chunk_pad_sec": 0.2,  # 덩어리 앞뒤 여유
    "chunk_max_sec": 30.0,  # 덩어리 최대 길이 (Whisper 창 길이)
    "diarization": "none",  # none / pyannote
    "merge_gap_sec": 1.0,  # 같은 화자 단어 사이가 이 이하면 같은 줄
    "max_line_sec": 30.0,  # 한 줄이 이보다 길어지면 끊는다
}


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """asr 단계를 실행한다.

    Args:
        video: 원본 영상 경로 (쓰지 않음. 오디오는 preprocess 결과를 쓴다).
        work_dir: 회차 작업 디렉터리. work/preprocess.json 과 work/audio.wav 가 있어야 한다.
        params: DEFAULT_PARAMS 를 덮어쓸 설정.

    Returns:
        work/asr.json 경로.

    Raises:
        RuntimeError: 영상에 오디오 트랙이 없음.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    pre = load_stage(work_dir, "preprocess", PreprocessResult).result
    if pre.audio_file is None:
        raise RuntimeError("오디오 트랙이 없어 asr 을 할 수 없습니다")
    audio = work_dir / pre.audio_file.path

    samples = load_wav(audio)
    model = load_whisper(p["model"], p["device"], p["compute_type"])
    segments = transcribe(model, samples, p)
    turns = diarize(audio, p["diarization"])
    if turns:
        lines = lines_from_words(segments, turns, p["merge_gap_sec"], p["max_line_sec"])
    else:
        lines = lines_from_segments(segments)
    return save_stage(work_dir, "asr", AsrResult(lines=lines), p)


SAMPLE_RATE = 16000


def load_wav(path: Path) -> np.ndarray:
    """preprocess 가 만든 16kHz 모노 16bit wav 를 float32 배열로 읽는다.

    faster-whisper 의 decode_audio(PyAV) 는 PyAV 버전에 따라 깨져서 쓰지 않는다 (av 19 에서 metadata_errors 인자 오류).

    Raises:
        ValueError: 16kHz 모노 16bit 가 아님.
    """
    with wave.open(str(path)) as w:
        if (w.getframerate(), w.getnchannels(), w.getsampwidth()) != (SAMPLE_RATE, 1, 2):
            raise ValueError(f"16kHz 모노 16bit wav 가 아님: {path}")
        data = w.readframes(w.getnframes())
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def load_whisper(size: str, device: str, compute_type: str):
    """faster-whisper 모델을 불러온다.

    모델은 Hugging Face 캐시(HF_HOME)에 받아 둔다. 워크스테이션에서는 HF_HOME 을 볼륨으로 잡아 두면 다시 받지 않는다.

    Args:
        size: 모델 크기 또는 경로 (small, medium, large-v3 …).
        device: auto / cpu / cuda.
        compute_type: default 면 cuda 는 float16, cpu 는 int8.

    Returns:
        WhisperModel.
    """
    from faster_whisper import WhisperModel

    if compute_type == "default":
        compute_type = "int8" if device == "cpu" else "default"
    return WhisperModel(size, device=device, compute_type=compute_type)


def speech_chunks(samples: np.ndarray, p: dict) -> list[tuple[int, int]]:
    """Silero VAD 로 말소리 덩어리를 나눈다.

    Args:
        samples: 16kHz 모노 float32.
        p: 설정 (vad_min_silence_ms, chunk_pad_sec, chunk_max_sec).

    Returns:
        (시작 샘플, 끝 샘플) 목록. 앞뒤로 chunk_pad_sec 여유를 둔다.
    """
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    opts = VadOptions(min_silence_duration_ms=p["vad_min_silence_ms"], max_speech_duration_s=p["chunk_max_sec"])
    pad = int(p["chunk_pad_sec"] * SAMPLE_RATE)
    out = []
    for ts in get_speech_timestamps(samples, opts):
        a = max(0, ts["start"] - pad)
        if out and a < out[-1][1]:
            a = out[-1][1]
        out.append((a, min(len(samples), ts["end"] + pad)))
    return out


def detect_language(model, chunk: np.ndarray, p: dict) -> tuple[str, float]:
    """덩어리의 언어를 후보 언어(languages) 안에서 고른다.

    대부분 대사는 default_language 이므로, 다른 언어로 바꾸는 것은 확률이 language_min_prob 이상일 때만 한다.
    (짧은 덩어리나 억양이 특이한 목소리는 다른 언어로 잘못 감지되기 쉽다. 합성 음성 테스트에서
    한국어 대사가 ja 0.25~0.68 로 나왔고, 실제 일본어 대사는 0.97~0.99 였다.)

    Returns:
        (언어, 확률). 짧거나 불확실하면 (default_language, 확률 또는 0.0).
    """
    if len(chunk) < p["min_detect_sec"] * SAMPLE_RATE:
        return p["default_language"], 0.0
    _lang, _prob, all_probs = model.detect_language(chunk)
    cands = [(lang, prob) for lang, prob in all_probs if lang in p["languages"]]
    if not cands:
        return p["default_language"], 0.0
    lang, prob = max(cands, key=lambda x: x[1])
    if lang != p["default_language"] and prob < p["language_min_prob"]:
        return p["default_language"], float(prob)
    return lang, float(prob)


def transcribe(model, samples: np.ndarray, p: dict) -> list[dict]:
    """말소리 덩어리마다 언어를 감지하고 그 언어로 받아 적는다.

    Args:
        model: load_whisper() 결과.
        samples: load_wav() 결과 (16kHz 모노 float32).
        p: 설정.

    Returns:
        Whisper 구간 목록 (시간순, 전체 기준 초).
        각 {"start", "end", "text", "language", "language_prob", "words": [{"start", "end", "text", "raw", "prob"}]}.
        raw 는 Whisper 가 준 조각 그대로 (앞 공백 포함). 일본어처럼 띄어쓰기 없는 언어도 그대로 이어 붙일 수 있다.
    """
    out = []
    for a, b in speech_chunks(samples, p):
        chunk = samples[a:b]
        offset = a / SAMPLE_RATE
        lang, lang_prob = detect_language(model, chunk, p)
        segs, _info = model.transcribe(
            chunk,
            language=lang,
            beam_size=p["beam_size"],
            vad_filter=False,  # 이미 VAD 로 나눈 덩어리
            word_timestamps=True,
            condition_on_previous_text=False,  # 앞 대사에 끌려 엉뚱한 문장을 반복하는 것을 줄인다
        )
        for s in segs:
            words = [{"start": offset + float(w.start), "end": offset + float(w.end), "text": w.word.strip(),
                      "raw": w.word, "prob": float(w.probability), "language": lang}
                     for w in (s.words or []) if w.word.strip()]
            out.append({"start": offset + float(s.start), "end": offset + float(s.end), "text": s.text.strip(),
                        "language": lang, "language_prob": round(lang_prob, 3), "words": words})
    return out


def diarize(audio: Path, backend: str) -> list[tuple[float, float, str]]:
    """화자 구분. 구간마다 S1, S2 … 라벨을 붙인다 (처음 말한 순서대로 번호).

    Args:
        audio: 16kHz 모노 wav 경로.
        backend: "none" 또는 "pyannote".

    Returns:
        (시작, 끝, 라벨) 목록. backend 가 none 이면 빈 목록.

    Raises:
        RuntimeError: pyannote 인데 HF_TOKEN 이 없음.
        ValueError: 알 수 없는 backend.
    """
    if backend == "none":
        return []
    if backend != "pyannote":
        raise ValueError(f"unknown diarization backend: {backend}")

    # 아직 실행해 보지 않은 코드. pyannote.audio 4.x 기준 (3.x 의 Annotation 반환도 처리)
    from pyannote.audio import Pipeline

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("pyannote 화자 구분에는 HF_TOKEN 환경변수가 필요합니다")
    pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-community-1", token=token)
    try:
        import torch

        if torch.cuda.is_available():
            pipeline.to(torch.device("cuda"))
    except ImportError:
        pass
    output = pipeline(str(audio))
    annotation = getattr(output, "speaker_diarization", output)
    raw = [(float(turn.start), float(turn.end), str(label))
           for turn, _, label in annotation.itertracks(yield_label=True)]
    raw.sort()
    names: dict[str, str] = {}
    for _, _, label in raw:
        names.setdefault(label, f"S{len(names) + 1}")
    return [(a, b, names[label]) for a, b, label in raw]


def _speaker_for(start: float, end: float, turns: list[tuple[float, float, str]]) -> str | None:
    """[start, end] 와 가장 많이 겹치는 화자. 겹치는 구간이 없으면 가장 가까운 구간의 화자."""
    best, best_overlap = None, 0.0
    for a, b, label in turns:
        overlap = min(end, b) - max(start, a)
        if overlap > best_overlap:
            best, best_overlap = label, overlap
    if best is not None:
        return best
    mid = (start + end) / 2
    nearest = min(turns, key=lambda t: min(abs(mid - t[0]), abs(mid - t[1])), default=None)
    return nearest[2] if nearest else None


def lines_from_words(segments: list[dict], turns: list[tuple[float, float, str]],
                     merge_gap: float, max_line: float) -> list[Line]:
    """단어마다 화자를 붙이고, 같은 화자가 이어서 말한 단어를 한 줄로 묶는다.

    Args:
        segments: transcribe() 결과.
        turns: diarize() 결과.
        merge_gap: 같은 화자 단어 사이가 이 이하면 같은 줄 (초).
        max_line: 한 줄 최대 길이 (초). 넘으면 끊는다.

    Returns:
        시간순 Line 목록 (id 는 1부터).
    """
    words = [w for s in segments for w in s["words"]]
    lines: list[Line] = []
    cur: list[dict] = []
    cur_spk: str | None = None

    def flush():
        if cur:
            lines.append(Line(id=len(lines) + 1, start_sec=round(cur[0]["start"], 3), end_sec=round(cur[-1]["end"], 3),
                              speaker=cur_spk, text=_join(w["raw"] for w in cur),
                              language=Counter(w["language"] for w in cur).most_common(1)[0][0],
                              words=[Word(start_sec=round(w["start"], 3), end_sec=round(max(w["end"], w["start"]), 3),
                                          text=w["text"], prob=round(w["prob"], 3)) for w in cur]))

    for w in words:
        spk = _speaker_for(w["start"], w["end"], turns)
        if cur and (spk != cur_spk or w["start"] - cur[-1]["end"] > merge_gap or w["end"] - cur[0]["start"] > max_line):
            flush()
            cur = []
        if not cur:
            cur_spk = spk
        cur.append(w)
    flush()
    return lines


def lines_from_segments(segments: list[dict]) -> list[Line]:
    """화자 구분이 없을 때: Whisper 구간을 그대로 한 줄씩 쓴다.

    Args:
        segments: transcribe() 결과.

    Returns:
        시간순 Line 목록 (speaker = None).
    """
    lines = []
    for s in segments:
        if not s["text"]:
            continue
        lines.append(Line(id=len(lines) + 1, start_sec=round(s["start"], 3), end_sec=round(max(s["end"], s["start"]), 3),
                          speaker=None, text=s["text"], language=s["language"],
                          words=[Word(start_sec=round(w["start"], 3), end_sec=round(max(w["end"], w["start"]), 3),
                                      text=w["text"], prob=round(w["prob"], 3)) for w in s["words"]]))
    return lines


def _join(tokens) -> str:
    """Whisper 단어 조각(앞 공백 포함)을 그대로 이어 문장을 만든다. 띄어쓰기는 Whisper 출력을 따른다."""
    return "".join(tokens).strip()
