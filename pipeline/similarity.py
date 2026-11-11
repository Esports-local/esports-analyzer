"""similarity 단계: 인접 segment 가 같은 장면으로 이어지는지 점수를 매긴다 (scenes 단계 입력).

점수
- visual: 화면 유사도. segment 마다 대표 프레임을 뽑아 이미지 임베딩(open_clip)의 평균을 내고,
  앞 segment 의 끝쪽 프레임과 뒤 segment 의 앞쪽 프레임을 비교한다 (컷 직전·직후 화면이 가장 직접적인 단서).
- dialogue: 대사가 경계를 넘으면 1.0, 아니면 0.0. asr 결과가 없으면 None.
  같은 화자가 경계 앞뒤로 말하면 0.5 (화자 구분이 있을 때만).
- audio: 아직 계산하지 않음 (None).

이미지 모델은 open_clip 으로 바꿔 끼울 수 있다 (params: clip_model, clip_pretrained).
설계안의 MobileCLIP 은 open_clip 에서의 정확한 모델 이름·라이선스를 워크스테이션에서 확인한 뒤 바꾼다.

입력: work/segments.json, work/asr.json (있으면)
출력: work/similarity.json
"""

from pathlib import Path

import numpy as np

from pipeline.io import load_stage, save_stage
from pipeline.media import sample_segment_frames
from pipeline.schema import AsrResult, Segment, SegmentPair, SegmentsResult, SimilarityResult

DEFAULT_PARAMS = {
    "clip_model": "ViT-B-32",
    "clip_pretrained": "laion2b_s34b_b79k",
    "device": "auto",  # auto / cpu / cuda
    "sample_fps": 2.0,  # segment 안에서 프레임을 뽑는 간격
    "frame_height": 224,
    "edge_frames": 3,  # 경계 비교에 쓸 segment 끝쪽/앞쪽 프레임 수
    "batch_size": 64,
}


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    """similarity 단계를 실행한다.

    Args:
        video: 원본 영상 경로.
        work_dir: 회차 작업 디렉터리. work/segments.json 이 있어야 한다 (work/asr.json 은 선택).
        params: DEFAULT_PARAMS 를 덮어쓸 설정.

    Returns:
        work/similarity.json 경로.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    segments = load_stage(work_dir, "segments", SegmentsResult).result.segments
    try:
        lines = load_stage(work_dir, "asr", AsrResult).result.lines
    except FileNotFoundError:
        lines = None

    embedder = ClipEmbedder(p["clip_model"], p["clip_pretrained"], p["device"])
    per_seg = segment_embeddings(video, segments, embedder, p)
    pairs = []
    for prev, nxt in zip(segments, segments[1:]):
        pairs.append(SegmentPair(
            prev_id=prev.id, next_id=nxt.id,
            visual=round(boundary_visual(per_seg.get(prev.id), per_seg.get(nxt.id), p["edge_frames"]), 4),
            dialogue=None if lines is None else dialogue_continuity(prev, nxt, lines),
            audio=None,
        ))
    return save_stage(work_dir, "similarity", SimilarityResult(pairs=pairs), p)


class ClipEmbedder:
    """open_clip 이미지 임베딩. 정규화된 벡터를 돌려준다."""

    def __init__(self, model_name: str, pretrained: str, device: str = "auto"):
        """모델을 불러온다.

        Args:
            model_name: open_clip 모델 이름 (예: ViT-B-32).
            pretrained: 사전학습 태그 (예: laion2b_s34b_b79k).
            device: auto / cpu / cuda.
        """
        import open_clip
        import torch

        self.torch = torch
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model = self.model.to(self.device).eval()

    def embed(self, frames_rgb: list[np.ndarray]) -> np.ndarray:
        """RGB 프레임 목록을 임베딩한다.

        Args:
            frames_rgb: (H, W, 3) uint8 RGB 배열 목록.

        Returns:
            (N, D) float32, 행마다 L2 정규화.
        """
        from PIL import Image

        if not frames_rgb:
            return np.zeros((0, 1), np.float32)
        batch = self.torch.stack([self.preprocess(Image.fromarray(f)) for f in frames_rgb]).to(self.device)
        with self.torch.no_grad():
            feats = self.model.encode_image(batch).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


def segment_embeddings(video: Path, segments: list[Segment], embedder: ClipEmbedder, p: dict) -> dict[int, np.ndarray]:
    """segment 마다 프레임 임베딩을 시간순으로 모은다.

    Args:
        video: 영상 경로.
        segments: segments 단계 결과.
        embedder: ClipEmbedder.
        p: 설정 (sample_fps, frame_height, batch_size).

    Returns:
        {segment id: (프레임 수, D) 임베딩}. 프레임을 하나도 못 뽑은 segment 는 빠진다.
    """
    ids: list[int] = []
    frames: list[np.ndarray] = []
    out: dict[int, list[np.ndarray]] = {}

    def flush():
        if frames:
            emb = embedder.embed(frames)
            for sid, e in zip(ids, emb):
                out.setdefault(sid, []).append(e)
            ids.clear()
            frames.clear()

    for seg, _t, bgr in sample_segment_frames(video, segments, p["sample_fps"], p["frame_height"]):
        ids.append(seg.id)
        frames.append(np.ascontiguousarray(bgr[..., ::-1]))  # BGR → RGB
        if len(frames) >= p["batch_size"]:
            flush()
    flush()
    return {sid: np.stack(v) for sid, v in out.items()}


def boundary_visual(prev: np.ndarray | None, nxt: np.ndarray | None, edge: int) -> float:
    """앞 segment 끝쪽 프레임 평균과 뒤 segment 앞쪽 프레임 평균의 코사인 유사도.

    Args:
        prev: 앞 segment 임베딩 (프레임 수, D). 없으면 None.
        nxt: 뒤 segment 임베딩. 없으면 None.
        edge: 양쪽에서 쓸 프레임 수.

    Returns:
        -1 ~ 1. 어느 한쪽이 없으면 0.0.
    """
    if prev is None or nxt is None or len(prev) == 0 or len(nxt) == 0:
        return 0.0
    a = prev[-edge:].mean(axis=0)
    b = nxt[:edge].mean(axis=0)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def dialogue_continuity(prev: Segment, nxt: Segment, lines) -> float:
    """두 segment 경계에서 대사가 이어지는지.

    - 대사 한 줄이 경계를 넘으면 1.0
    - 경계 직전·직후 대사의 화자가 같으면 0.5 (화자 구분이 있을 때만)
    - 그 외 0.0

    Args:
        prev: 앞 segment.
        nxt: 뒤 segment.
        lines: asr 결과 대사 줄.

    Returns:
        0.0, 0.5, 1.0 중 하나.
    """
    cut_a, cut_b = prev.end_sec, nxt.start_sec
    for ln in lines:
        if ln.start_sec < cut_a - 0.05 and ln.end_sec > cut_b + 0.05:
            return 1.0
    before = [ln for ln in lines if ln.end_sec <= cut_a + 0.05 and ln.end_sec > cut_a - 3.0]
    after = [ln for ln in lines if ln.start_sec >= cut_b - 0.05 and ln.start_sec < cut_b + 3.0]
    if before and after and before[-1].speaker and before[-1].speaker == after[0].speaker:
        return 0.5
    return 0.0
