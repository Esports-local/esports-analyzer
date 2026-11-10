"""faces 단계: InsightFace(buffalo_l) 로 얼굴을 찾고 회차 안에서 같은 사람끼리 묶는다.

1. segments.json 의 segment 구간에서 sample_fps 로 프레임을 뽑는다 (전환 구간은 건너뜀).
   sample_fps 간격보다 짧은 segment 는 가운데 프레임을 하나 따로 뽑는다.
2. 얼굴 검출(SCRFD) → 정렬 → 임베딩(ArcFace, 512차원 정규화).
   det_score < min_det_score 이거나 얼굴 높이 < min_face_px 이면 버린다.
3. 코사인 유사도로 군집화: 검출 점수 높은 얼굴부터 가장 가까운 군집에 넣고(유사도 ≥ cluster_threshold),
   없으면 새 군집. 이후 중심끼리 가까운 군집을 합친다.
   얼굴 수 < min_cluster_size 인 군집은 라벨 없음(None). 라벨은 얼굴 수가 많은 순서로 F1, F2 …

모델: buffalo_l 사전학습 가중치는 비상업 연구용 라이선스.
"""

import os
import subprocess
from pathlib import Path

import numpy as np

from pipeline.io import load_stage, save_stage
from pipeline.schema import FaceCluster, FaceDetection, FacesResult, Segment, SegmentsResult

DEFAULT_PARAMS = {
    "model": "insightface/buffalo_l",
    "sample_fps": 2.0,
    "frame_height": None,  # None 이면 원본 해상도 (작은 얼굴까지 잡기 위해)
    "det_size": 640,
    "min_det_score": 0.6,
    "min_face_px": 40,
    "cluster_threshold": 0.45,
    "min_cluster_size": 3,
}


def run(video: Path, work_dir: Path, params: dict | None = None) -> Path:
    p = {**DEFAULT_PARAMS, **(params or {})}
    segments = load_stage(work_dir, "segments", SegmentsResult).result.segments
    app = load_model(p["det_size"])

    detections, embeddings = [], []
    for seg, t, frame in sample_frames(video, segments, p["sample_fps"], p["frame_height"]):
        h, w = frame.shape[:2]
        for face in app.get(frame):
            x1, y1, x2, y2 = (float(v) for v in face.bbox)
            if face.det_score < p["min_det_score"] or y2 - y1 < p["min_face_px"]:
                continue
            bbox = (max(x1 / w, 0.0), max(y1 / h, 0.0), min(x2 / w, 1.0), min(y2 / h, 1.0))
            detections.append(FaceDetection(id=len(detections) + 1, segment_id=seg.id, time_sec=round(t, 3),
                                            bbox=tuple(round(v, 4) for v in bbox),
                                            det_score=round(float(face.det_score), 3), face_id=None))
            embeddings.append(face.normed_embedding.astype(np.float32))

    emb = np.stack(embeddings) if embeddings else np.zeros((0, 512), np.float32)
    labels = cluster(emb, np.array([d.det_score for d in detections]), p["cluster_threshold"], p["min_cluster_size"])
    clusters = []
    for label in sorted({l for l in labels if l is not None}):
        idx = [i for i, l in enumerate(labels) if l == label]
        face_id = f"F{label}"
        for i in idx:
            detections[i].face_id = face_id
        best = max(idx, key=lambda i: detections[i].det_score)
        clusters.append(FaceCluster(face_id=face_id, count=len(idx),
                                    segment_ids=sorted({detections[i].segment_id for i in idx}),
                                    best_detection_id=detections[best].id))

    emb_path = work_dir / "work" / "faces_embeddings.npy"
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, emb)
    result = FacesResult(embeddings_path=str(emb_path.relative_to(work_dir)), clusters=clusters, detections=detections)
    return save_stage(work_dir, "faces", result, p)


def load_model(det_size: int):
    import warnings

    from insightface.app import FaceAnalysis

    warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

    app = FaceAnalysis(name="buffalo_l", root=os.environ.get("INSIGHTFACE_ROOT", "~/.insightface"),
                       allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def _segment_at(segments: list[Segment], t: float) -> Segment | None:
    for s in segments:
        if s.start_sec <= t < s.end_sec:
            return s
    return None


def _frame_size(video: Path, height: int | None) -> tuple[int, int]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "csv=p=0", str(video)],
                         check=True, capture_output=True, text=True).stdout.strip().split(",")
    w, h = int(out[0]), int(out[1])
    if height is None or height == h:
        return (w, h)
    return (round(w * height / h / 2) * 2, height)


def sample_frames(video: Path, segments: list[Segment], fps: float, height: int | None):
    """(segment, 시각, BGR 프레임) 을 순서대로 낸다."""
    w, h = _frame_size(video, height)
    size = w * h * 3
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(video), "-an",
                             "-vf", f"fps={fps},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:"],
                            stdout=subprocess.PIPE)
    covered: set[int] = set()
    k = 0
    while True:
        buf = proc.stdout.read(size)
        if len(buf) < size:
            break
        t = k / fps
        k += 1
        seg = _segment_at(segments, t)
        if seg is not None:
            covered.add(seg.id)
            yield seg, t, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    proc.wait()

    for seg in segments:  # 샘플이 하나도 없던 짧은 segment 는 가운데 프레임
        if seg.id in covered:
            continue
        t = (seg.start_sec + seg.end_sec) / 2
        raw = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", str(video), "-frames:v", "1",
                              "-vf", f"scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:"],
                             check=True, capture_output=True).stdout
        if len(raw) == size:
            yield seg, t, np.frombuffer(raw, np.uint8).reshape(h, w, 3)


def cluster(emb: np.ndarray, scores: np.ndarray, threshold: float, min_size: int) -> list[int | None]:
    """얼굴 임베딩 군집화. 반환: 얼굴마다 군집 번호(1부터, 큰 군집 순) 또는 None."""
    n = len(emb)
    if n == 0:
        return []
    assign = np.full(n, -1)
    sums: list[np.ndarray] = []
    for i in np.argsort(-scores):
        if sums:
            cent = np.stack([s / np.linalg.norm(s) for s in sums])
            sims = cent @ emb[i]
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                assign[i] = j
                sums[j] = sums[j] + emb[i]
                continue
        assign[i] = len(sums)
        sums.append(emb[i].copy())

    # 중심이 가까운 군집 합치기
    merged = True
    while merged and len(sums) > 1:
        merged = False
        cent = np.stack([s / np.linalg.norm(s) for s in sums])
        sim = cent @ cent.T
        np.fill_diagonal(sim, -1)
        a, b = np.unravel_index(np.argmax(sim), sim.shape)
        if sim[a, b] >= threshold:
            a, b = min(a, b), max(a, b)
            sums[a] = sums[a] + sums[b]
            sums.pop(b)
            assign[assign == b] = a
            assign[assign > b] -= 1
            merged = True

    counts = np.bincount(assign, minlength=len(sums))
    order = [c for c in np.argsort(-counts, kind="stable") if counts[c] >= min_size]
    rank = {int(c): r + 1 for r, c in enumerate(order)}
    return [rank.get(int(a)) for a in assign]
