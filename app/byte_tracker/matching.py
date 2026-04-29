import numpy as np
import lap
from scipy.spatial.distance import cdist

from . import kalman_filter


def merge_matches(m1, m2, shape):
    o, p, q = shape
    m1 = np.asarray(m1)
    m2 = np.asarray(m2)
    import scipy.sparse

    m1s = scipy.sparse.coo_matrix((np.ones(len(m1)), (m1[:, 0], m1[:, 1])), shape=(o, p))
    m2s = scipy.sparse.coo_matrix((np.ones(len(m2)), (m2[:, 0], m2[:, 1])), shape=(p, q))
    mask = m1s * m2s
    match = mask.nonzero()
    match = list(zip(match[0], match[1]))
    unmatched_o = tuple(set(range(o)) - set(i for i, _ in match))
    unmatched_q = tuple(set(range(q)) - set(j for _, j in match))
    return match, unmatched_o, unmatched_q


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return (
            np.empty((0, 2), dtype=int),
            tuple(range(cost_matrix.shape[0])),
            tuple(range(cost_matrix.shape[1])),
        )
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    matches = []
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches) if matches else np.empty((0, 2), dtype=int)
    return matches, unmatched_a, unmatched_b


def bbox_overlaps_xyxy(boxes: np.ndarray, query_boxes: np.ndarray) -> np.ndarray:
    """IoU matrix (N, K), xyxy format."""
    n = boxes.shape[0]
    k = query_boxes.shape[0]
    if n == 0 or k == 0:
        return np.zeros((n, k), dtype=np.float64)
    area1 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area2 = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1])
    lt = np.maximum(boxes[:, None, :2], query_boxes[None, :, :2])
    rb = np.minimum(boxes[:, None, 2:], query_boxes[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    return inter / np.clip(union, 1e-6, None)


def ious(atlbrs, btlbrs):
    if isinstance(atlbrs, list) and len(atlbrs) > 0 and not isinstance(atlbrs[0], np.ndarray):
        atlbrs = np.asarray([t for t in atlbrs], dtype=np.float64)
    if isinstance(btlbrs, list) and len(btlbrs) > 0 and not isinstance(btlbrs[0], np.ndarray):
        btlbrs = np.asarray([t for t in btlbrs], dtype=np.float64)
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float64)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float64)
    return bbox_overlaps_xyxy(atlbrs, btlbrs)


def iou_distance(atracks, btracks):
    if (len(atracks) > 0 and isinstance(atracks[0], np.ndarray)) or (
        len(btracks) > 0 and isinstance(btracks[0], np.ndarray)
    ):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = np.asarray([track.tlbr for track in atracks], dtype=np.float64)
        btlbrs = np.asarray([track.tlbr for track in btracks], dtype=np.float64)
    _ious = ious(atlbrs, btlbrs)
    return 1 - _ious


def embedding_distance(tracks, detections, metric="cosine"):
    cost_matrix = np.zeros((len(tracks), len(detections)), dtype=np.float64)
    if cost_matrix.size == 0:
        return cost_matrix
    det_features = np.asarray([track.curr_feat for track in detections], dtype=np.float64)
    track_features = np.asarray([track.smooth_feat for track in tracks], dtype=np.float64)
    return np.maximum(0.0, cdist(track_features, det_features, metric))


def gate_cost_matrix(kf, cost_matrix, tracks, detections, only_position=False):
    if cost_matrix.size == 0:
        return cost_matrix
    gating_dim = 2 if only_position else 4
    gating_threshold = kalman_filter.chi2inv95[gating_dim]
    measurements = np.asarray([det.to_xyah() for det in detections])
    for row, track in enumerate(tracks):
        gating_distance = kf.gating_distance(
            track.mean, track.covariance, measurements, only_position
        )
        cost_matrix[row, gating_distance > gating_threshold] = np.inf
    return cost_matrix


def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    return 1 - fuse_sim
