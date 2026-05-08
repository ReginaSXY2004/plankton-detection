from dataclasses import dataclass
from typing import List, Dict, Tuple
import math

try:
    from scipy.optimize import linear_sum_assignment
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def cx(self) -> float:
        return self.x1 + self.w / 2.0

    @property
    def cy(self) -> float:
        return self.y1 + self.h / 2.0


@dataclass
class Track:
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int

    cx: float
    cy: float
    vx: float = 0.0
    vy: float = 0.0

    hits: int = 1
    missed: int = 0
    age: int = 1

    _VEL_WINDOW: int = 4

    def __post_init__(self):
        self._vx_history: list = []
        self._vy_history: list = []
        self._prev_cx = self.cx
        self._prev_cy = self.cy
        self._stationary_frames = 0
        self._occluded = False

    def update_from_detection(self, det: "Detection"):
        old_cx, old_cy = self.cx, self.cy
        new_cx, new_cy = det.cx, det.cy

        raw_vx = new_cx - old_cx
        raw_vy = new_cy - old_cy

        self._vx_history.append(raw_vx)
        self._vy_history.append(raw_vy)
        if len(self._vx_history) > self._VEL_WINDOW:
            self._vx_history.pop(0)
            self._vy_history.pop(0)

        self.vx = sum(self._vx_history) / len(self._vx_history)
        self.vy = sum(self._vy_history) / len(self._vy_history)

        speed_now = math.hypot(raw_vx, raw_vy)
        if speed_now < 1.5:
            self._stationary_frames += 1
        else:
            self._stationary_frames = 0

        self._prev_cx = self.cx
        self._prev_cy = self.cy

        self.x1, self.y1, self.x2, self.y2 = det.x1, det.y1, det.x2, det.y2
        self.conf = det.conf
        self.cls_id = det.cls_id
        self.cx, self.cy = new_cx, new_cy

        self.hits += 1
        self.missed = 0
        self.age += 1
        self._occluded = False

    def mark_missed(self):
        self.missed += 1
        self.age += 1

        speed = math.hypot(self.vx, self.vy)

        # 静止目标：尽量原地等，不要漂
        if self._stationary_frames >= 2 or speed < 1.5:
            decay = 0.0
        # 慢速目标：轻微衰减外推
        elif speed < 6.0:
            decay = 0.25
        # 正常目标：适度外推
        else:
            decay = 0.50

        self.cx += self.vx * decay
        self.cy += self.vy * decay

    @property
    def w(self) -> float:
        return self.x2 - self.x1

    @property
    def h(self) -> float:
        return self.y2 - self.y1

    @property
    def bbox(self):
        w, h = self.w, self.h
        x1 = self.cx - w / 2.0
        y1 = self.cy - h / 2.0
        x2 = self.cx + w / 2.0
        y2 = self.cy + h / 2.0
        return x1, y1, x2, y2

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


class MicrobeTracker:
    """
    针对可动微生物的轻量 tracker，重点解决：
    1. 重复计数：优先接回 lost track，而不是在旁边新建 ID
    2. ID 交换：匹配 cost 不只看距离，还看 IoU / 方向 / 尺寸
    3. 静止漏跟：静止目标 missed 时尽量原地等待，不乱漂
    """

    def __init__(
        self,
        max_missing: int = 12,
        min_hits_to_show: int = 5,
        base_distance_thresh: float = 25.0,
        distance_scale: float = 1.5,
        max_size_ratio: float = 2.2,
        conf_threshold_for_tracking: float = 0.35,
        max_speed_px: float = 60.0,
        speed_scale: float = 3.0,
        no_spawn_radius: float = 30.0,
        reconnect_max_missing: int = 10,
        debug_print: bool = False,
    ):
        self.max_missing = max_missing
        self.min_hits_to_show = min_hits_to_show
        self.base_distance_thresh = base_distance_thresh
        self.distance_scale = distance_scale
        self.max_size_ratio = max_size_ratio
        self.conf_threshold_for_tracking = conf_threshold_for_tracking
        self.max_speed_px = max_speed_px
        self.speed_scale = speed_scale
        self.no_spawn_radius = no_spawn_radius
        self.reconnect_max_missing = reconnect_max_missing
        self.debug_print = debug_print

        self.tracks: Dict[int, Track] = {}
        self.next_id = 1
    def _try_reconnect_lost_track(self, det: Detection):
        best_tid = None
        best_cost = 1e9

        for tid, tr in self.tracks.items():
            # 只考虑刚丢失不久的轨迹
            if tr.missed < 1 or tr.missed > self.reconnect_max_missing:
                continue

            # 框变化太大就不接
            if self._size_ratio(tr, det) > min(self.max_size_ratio, 2.0):
                continue

            dist = self._center_distance(tr.cx, tr.cy, det.cx, det.cy)
            reconnect_thresh = max(
                self.base_distance_thresh * 2,
                max(tr.w, tr.h, det.w, det.h) * (self.distance_scale+0.2)
            )

            if dist > reconnect_thresh:
                continue

            cost = self._match_cost(tr, det)
            if cost < best_cost:
                best_cost = cost
                best_tid = tid

        return best_tid
    @staticmethod
    def _center_distance(cx1: float, cy1: float, cx2: float, cy2: float) -> float:
        return math.hypot(cx1 - cx2, cy1 - cy2)

    @staticmethod
    def _bbox_iou(box1, box2) -> float:
        x11, y11, x12, y12 = box1
        x21, y21, x22, y22 = box2

        ix1 = max(x11, x21)
        iy1 = max(y11, y21)
        ix2 = min(x12, x22)
        iy2 = min(y12, y22)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        a1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
        a2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
        union = a1 + a2 - inter
        if union <= 1e-6:
            return 0.0
        return inter / union

    @staticmethod
    def _size_ratio(track: Track, det: Detection) -> float:
        tw = max(track.w, 1e-6)
        th = max(track.h, 1e-6)
        dw = max(det.w, 1e-6)
        dh = max(det.h, 1e-6)
        ratio_w = max(tw / dw, dw / tw)
        ratio_h = max(th / dh, dh / th)
        return max(ratio_w, ratio_h)

    def _prune_overlapping_visible_tracks(self, tracks: List[Track]) -> List[Track]:
        tracks = sorted(
            tracks,
            key=lambda t: (t.hits, t.conf, -t.missed),
            reverse=True
        )

        kept = []
        for tr in tracks:
            keep = True
            for k in kept:
                iou = self._bbox_iou(tr.bbox, k.bbox)
                d = self._center_distance(tr.cx, tr.cy, k.cx, k.cy)

                if iou > 0.55 or d < 15:
                    keep = False
                    break

            if keep:
                kept.append(tr)

        return kept

    def _distance_threshold(self, track: Track, det: Detection) -> float:
        dynamic = max(track.w, track.h, det.w, det.h) * self.distance_scale
        return max(self.base_distance_thresh, dynamic)

    def _speed_threshold(self, track: Track) -> float:
        if track.missed > 0:
            return self.max_speed_px * min(track.missed + 1, 3)

        if track.hits <= 3:
            return self.max_speed_px

        speed_based = track.speed * self.speed_scale
        return min(self.max_speed_px, max(speed_based, self.base_distance_thresh))

    def _can_match(self, track: Track, det: Detection) -> bool:
        if det.conf < self.conf_threshold_for_tracking:
            return False

        if self._size_ratio(track, det) > self.max_size_ratio:
            return False

        dist = self._center_distance(track.cx, track.cy, det.cx, det.cy)
        dist_thresh = self._distance_threshold(track, det)
        speed_thresh = self._speed_threshold(track)

        if track.missed == 0:
            effective_thresh = min(dist_thresh, speed_thresh)
        else:
            effective_thresh = max(dist_thresh, speed_thresh)

        return dist <= effective_thresh

    def _direction_penalty(self, track: Track, det: Detection) -> float:
        move_x = det.cx - track.cx
        move_y = det.cy - track.cy
        move_norm = math.hypot(move_x, move_y)
        vel_norm = math.hypot(track.vx, track.vy)

        # 速度很小，不加方向惩罚
        if move_norm < 1e-6 or vel_norm < 1.0:
            return 0.0

        cos_sim = (move_x * track.vx + move_y * track.vy) / (move_norm * vel_norm + 1e-6)
        cos_sim = max(-1.0, min(1.0, cos_sim))

        # 同方向 ~ 0，反方向 ~ 2
        return 1.0 - cos_sim

    def _match_cost(self, track: Track, det: Detection) -> float:
        dist = self._center_distance(track.cx, track.cy, det.cx, det.cy)
        dist_thresh = max(self._distance_threshold(track, det), 1e-6)
        dist_cost = dist / dist_thresh

        iou = self._bbox_iou(track.bbox, (det.x1, det.y1, det.x2, det.y2))
        iou_cost = 1.0 - iou

        size_cost = min(self._size_ratio(track, det) - 1.0, 2.0)
        dir_cost = self._direction_penalty(track, det)

        # lost track 优先允许接回，不要对它太苛刻
        missed_bonus = 0.15 * min(track.missed, 2)

        cost = (
            0.55 * dist_cost +
            0.25 * iou_cost +
            0.12 * dir_cost +
            0.08 * size_cost
        ) - missed_bonus

        return cost

    def _too_close_to_existing_track(self, det: Detection) -> bool:
        for tr in self.tracks.values():
            # 不只看活跃轨迹，也看短暂丢失的轨迹，避免旧 ID 旁边重新出生
            if tr.missed > min(self.max_missing, 6):
                continue

            d = self._center_distance(tr.cx, tr.cy, det.cx, det.cy)
            dynamic_radius = max(
                self.no_spawn_radius,
                0.35 * max(tr.w, tr.h, det.w, det.h),
                0.25 * (tr.w + tr.h)
            )
            if d < dynamic_radius:
                return True
        return False

    def _solve_assignment(
        self,
        track_ids: List[int],
        detections: List[Detection]
    ) -> List[Tuple[int, int]]:
        if not track_ids or not detections:
            return []

        INF = 1e6
        cost_matrix = []

        for tid in track_ids:
            tr = self.tracks[tid]
            row = []
            for det in detections:
                if self._can_match(tr, det):
                    row.append(self._match_cost(tr, det))
                else:
                    row.append(INF)
            cost_matrix.append(row)

        matches = []

        if _HAS_SCIPY:
            import numpy as np
            cm = np.array(cost_matrix, dtype=float)
            row_ind, col_ind = linear_sum_assignment(cm)
            for r, c in zip(row_ind, col_ind):
                if cm[r, c] < INF * 0.5:
                    matches.append((track_ids[r], c))
            return matches

        # fallback：没有 scipy 时，用改良贪心
        candidates = []
        for r, tid in enumerate(track_ids):
            for c in range(len(detections)):
                cost = cost_matrix[r][c]
                if cost < INF * 0.5:
                    candidates.append((cost, tid, c))

        # 更稳定一点：先按 cost，再按 hits 倒序，再按 missed 升序
        candidates.sort(
            key=lambda x: (
                x[0],
                -self.tracks[x[1]].hits,
                self.tracks[x[1]].missed
            )
        )

        used_tracks = set()
        used_dets = set()
        for cost, tid, di in candidates:
            if tid in used_tracks or di in used_dets:
                continue
            used_tracks.add(tid)
            used_dets.add(di)
            matches.append((tid, di))

        return matches

    def _mark_occlusion(self):
        tids = list(self.tracks.keys())
        for i in range(len(tids)):
            for j in range(i + 1, len(tids)):
                tr1 = self.tracks[tids[i]]
                tr2 = self.tracks[tids[j]]

                if tr1.missed > 1 or tr2.missed > 1:
                    continue

                d = self._center_distance(tr1.cx, tr1.cy, tr2.cx, tr2.cy)
                near_thresh = 0.6 * max(tr1.w, tr1.h, tr2.w, tr2.h)

                iou = self._bbox_iou(tr1.bbox, tr2.bbox)
                if d < near_thresh or iou > 0.15:
                    tr1._occluded = True
                    tr2._occluded = True

    def update(self, detections: List[Detection]) -> List[Track]:
        """
        1. 过滤低置信度 detection
        2. 拿出当前已有 tracks
        3. 匹配旧 track 和新 detection
        4. 匹配上的 track 更新位置
        5. 没匹配上的 track 标记 missed
        6. 没匹配上的 detection 尝试新建 track
        7. 返回稳定可见 tracks
        """
        detections = [d for d in detections if d.conf >= self.conf_threshold_for_tracking]

        track_ids = list(self.tracks.keys())
        unmatched_tracks = set(track_ids)
        unmatched_dets = set(range(len(detections)))

        matches = self._solve_assignment(track_ids, detections)

        for tid, di in matches:
            if tid not in self.tracks:
                continue
            self.tracks[tid].update_from_detection(detections[di])
            unmatched_tracks.discard(tid)
            unmatched_dets.discard(di)

        # 没匹配上的旧轨迹：miss + 预测
        for tid in list(unmatched_tracks):
            if tid in self.tracks:
                self.tracks[tid].mark_missed()

        # 交叉/靠近时打一下 occlusion 标记
        self._mark_occlusion()

        # 删除丢失太久的轨迹
        to_delete = [tid for tid, tr in self.tracks.items() if tr.missed > self.max_missing]
        for tid in to_delete:
            del self.tracks[tid]

        # 新建 track 前先检查 no-spawn zone
        for di in list(unmatched_dets):
            det = detections[di]

            reconnect_tid = self._try_reconnect_lost_track(det)
            if reconnect_tid is not None and reconnect_tid in self.tracks:
                self.tracks[reconnect_tid].update_from_detection(det)
                continue

            if self._too_close_to_existing_track(det):
                continue

            new_track = Track(
                track_id=self.next_id,
                x1=det.x1,
                y1=det.y1,
                x2=det.x2,
                y2=det.y2,
                conf=det.conf,
                cls_id=det.cls_id,
                cx=det.cx,
                cy=det.cy,
            )
            self.tracks[self.next_id] = new_track
            self.next_id += 1

        visible_tracks = [
            tr for tr in self.tracks.values()
            if tr.hits >= self.min_hits_to_show and tr.missed <= 1
        ]
        visible_tracks = self._prune_overlapping_visible_tracks(visible_tracks)
        if self.debug_print:
            print(f"active_tracks: {len(self.tracks)} | visible: {len(visible_tracks)}")
        return visible_tracks