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
from pathlib import Path

import numpy as np

from pipeline.io import load_stage, save_stage
from pipeline.media import sample_segment_frames
from pipeline.schema import FaceCluster, FaceDetection, FacesResult, SegmentsResult

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
    """faces 단계를 실행한다.

    segments 결과의 구간에서 프레임을 뽑아 얼굴을 찾고, 회차 안에서 같은 사람끼리 묶어 F1, F2 … 라벨을 붙인다.

    Args:
        video: 원본 영상 경로.
        work_dir: 회차 작업 디렉터리. work/segments.json 이 있어야 한다.
        params: DEFAULT_PARAMS 를 덮어쓸 설정.

    Returns:
        work/faces.json 경로. 임베딩은 work/faces_embeddings.npy 에 저장한다.
    """
    p = {**DEFAULT_PARAMS, **(params or {})}
    segments = load_stage(work_dir, "segments", SegmentsResult).result.segments
    app = load_model(p["det_size"])

    detections, embeddings = [], []
    for seg, t, frame in sample_segment_frames(video, segments, p["sample_fps"], p["frame_height"]):
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
    """InsightFace buffalo_l 의 검출·인식 모델만 CPU 로 불러온다.

    모델 파일은 INSIGHTFACE_ROOT (이미지 빌드 시 /opt/insightface) 에서 읽는다.

    Args:
        det_size: 검출 모델 입력 크기 (정사각형 한 변, px).

    Returns:
        준비된 FaceAnalysis. app.get(BGR 이미지) 로 얼굴 목록을 얻는다.
    """
    import warnings

    from insightface.app import FaceAnalysis

    warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")

    app = FaceAnalysis(name="buffalo_l", root=os.environ.get("INSIGHTFACE_ROOT", "~/.insightface"),
                       allowed_modules=["detection", "recognition"], providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(det_size, det_size))
    return app


def cluster(emb: np.ndarray, scores: np.ndarray, threshold: float, min_size: int) -> list[int | None]:
    """얼굴 임베딩을 코사인 유사도로 군집화한다.

    1. 검출 점수가 높은 얼굴부터, 가장 가까운 군집 중심과의 유사도가 threshold 이상이면 그 군집에 넣고
       아니면 새 군집을 만든다. 중심 행렬을 유지하고 바뀐 군집의 행만 갱신한다.
    2. 중심끼리 유사도가 threshold 이상인 군집을 가장 가까운 쌍부터 합친다.
       행마다 최대 유사도를 기억해 두고, 합칠 때 영향받는 행만 다시 계산한다.
       (군집 수 k 에 대해 k x k 유사도 행렬을 쓴다. k=5,000 이면 약 100MB)

    Args:
        emb: (N, D) 정규화된 임베딩.
        scores: (N,) 검출 점수.
        threshold: 같은 사람으로 볼 코사인 유사도.
        min_size: 이보다 얼굴 수가 적은 군집은 라벨을 붙이지 않는다.

    Returns:
        얼굴마다 군집 번호 (1부터, 얼굴 수가 많은 순) 또는 None.
    """
    n = len(emb)
    if n == 0:
        return []
    emb = emb.astype(np.float32, copy=False)
    assign = np.full(n, -1, dtype=np.int64)
    sums = np.zeros_like(emb)
    cent = np.zeros_like(emb)
    k = 0
    for i in np.argsort(-scores, kind="stable"):
        e = emb[i]
        if k:
            sims = cent[:k] @ e
            j = int(np.argmax(sims))
            if sims[j] >= threshold:
                assign[i] = j
                sums[j] += e
                cent[j] = sums[j] / np.linalg.norm(sums[j])
                continue
        assign[i] = k
        sums[k] = e
        cent[k] = e / np.linalg.norm(e)
        k += 1

    # 중심이 가까운 군집 합치기
    alive = np.ones(k, dtype=bool)
    sim = cent[:k] @ cent[:k].T
    np.fill_diagonal(sim, -np.inf)
    best_idx = sim.argmax(axis=1)
    best_val = sim[np.arange(k), best_idx]
    while k > 1:
        a = int(np.argmax(best_val))
        if best_val[a] < threshold:
            break
        b = int(best_idx[a])
        sums[a] += sums[b]
        cent[a] = sums[a] / np.linalg.norm(sums[a])
        assign[assign == b] = a
        alive[b] = False
        sim[b, :] = -np.inf
        sim[:, b] = -np.inf
        best_val[b] = -np.inf
        row = cent[:k] @ cent[a]
        row[~alive] = -np.inf
        row[a] = -np.inf
        sim[a, :] = row
        sim[:, a] = row
        # 최대값이 a 나 b 를 가리키던 행, 그리고 a 행 자신은 다시 계산
        stale = alive & ((best_idx == a) | (best_idx == b))
        stale[a] = True
        for r in np.flatnonzero(stale):
            best_idx[r] = int(np.argmax(sim[r]))
            best_val[r] = sim[r, best_idx[r]]
        # 새 a 열이 기존 최대보다 큰 행 갱신
        better = alive & (row > best_val)
        best_idx[better] = a
        best_val[better] = row[better]

    counts = np.bincount(assign, minlength=k)
    order = [c for c in np.argsort(-counts, kind="stable") if alive[c] and counts[c] >= min_size]
    rank = {int(c): r + 1 for r, c in enumerate(order)}
    return [rank.get(int(a)) for a in assign]
