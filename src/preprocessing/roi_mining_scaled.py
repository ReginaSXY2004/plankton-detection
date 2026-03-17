import cv2
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from preprocessing.background_subtraction import (
    create_background_subtractor,
    subtract_background,
)
from inference.roi_predictor import ROIPredictor

REFERENCE_MAGNIFICATION = 1.0
REFERENCE_FRAME_SIZE = (2448, 2048)  # width, height for the known 1.0x video

@dataclass
class ROIParams:
    warmup_frames: int = 30
    debug_save_until_frame: int = 35
    gray_threshold: int = 35

    # contour filters (defined at 1.0x reference scale)
    min_area_ref: float = 700.0
    max_area_ref: float = 20000.0
    min_w_ref: int = 25
    min_h_ref: int = 25
    max_w_ref: int = 300
    max_h_ref: int = 300
    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 6.5

    # edge logic
    edge_margin_ref: int = 3
    large_object_w_ref: int = 40
    large_object_h_ref: int = 40
    large_object_area_ref: float = 800.0
    allow_small_edge_objects: bool = True

    # tracker
    tracker_iou_threshold: float = 0.3
    tracker_max_missing: int = 45 #物体短暂离开视野/被遮挡时，等待更多帧再放弃这条 track，而不是马上新建一条
    roi_pad_ref: int = 18
    min_track_age: int = 2 # 新建的 track 必须连续存活这么多帧才算有效
    min_displacement_ref: float = 10

    # morphology kernels (defined at 1.0x reference scale)
    fill_kernel_ref: int = 15
    small_kernel_ref: int = 4
    merge_kernel_ref: int = 11

    # CNN switch
    enable_cnn: bool = True  # 设为 False 可跳过 CNN 推理，只保存 ROI 图像


MAGNIFICATION_PROFILES = {
    1.0: ROIParams(),
    0.5: ROIParams(),
    0.2: ROIParams(),
}


class ScaledParams:
    def __init__(self, base: ROIParams, magnification: float, frame_size: tuple[int, int]):
        frame_w, frame_h = frame_size
        ref_w, ref_h = REFERENCE_FRAME_SIZE

        mag_scale = magnification / REFERENCE_MAGNIFICATION
        res_scale_x = frame_w / ref_w
        res_scale_y = frame_h / ref_h
        length_scale = mag_scale * (res_scale_x + res_scale_y) / 2.0
        area_scale = max(length_scale ** 2, 1e-6)

        self.warmup_frames = base.warmup_frames
        self.debug_save_until_frame = base.debug_save_until_frame
        self.gray_threshold = base.gray_threshold
        self.min_aspect_ratio = base.min_aspect_ratio
        self.max_aspect_ratio = base.max_aspect_ratio
        self.tracker_iou_threshold = base.tracker_iou_threshold
        self.tracker_max_missing = base.tracker_max_missing
        self.min_track_age = base.min_track_age
        self.allow_small_edge_objects = base.allow_small_edge_objects
        self.enable_cnn = base.enable_cnn  # 透传 CNN 开关

        self.min_area = max(1, int(base.min_area_ref * area_scale))
        self.max_area = max(self.min_area + 1, int(base.max_area_ref * area_scale))
        self.min_w = max(4, int(round(base.min_w_ref * length_scale)))
        self.min_h = max(4, int(round(base.min_h_ref * length_scale)))
        self.max_w = max(self.min_w + 1, int(round(base.max_w_ref * length_scale)))
        self.max_h = max(self.min_h + 1, int(round(base.max_h_ref * length_scale)))

        self.edge_margin = max(1, int(round(base.edge_margin_ref * length_scale)))
        self.large_object_w = max(8, int(round(base.large_object_w_ref * length_scale)))
        self.large_object_h = max(8, int(round(base.large_object_h_ref * length_scale)))
        self.large_object_area = max(16, int(round(base.large_object_area_ref * area_scale)))

        self.roi_pad = max(2, int(round(base.roi_pad_ref * length_scale)))
        self.min_displacement = max(2.0, base.min_displacement_ref * length_scale)

        self.fill_kernel = ensure_odd(max(3, int(round(base.fill_kernel_ref * length_scale))))
        self.small_kernel = ensure_odd(max(3, int(round(base.small_kernel_ref * length_scale))))
        self.merge_kernel = ensure_odd(max(3, int(round(base.merge_kernel_ref * length_scale))))

        self.length_scale = length_scale
        self.area_scale = area_scale
        self.frame_size = frame_size
        self.magnification = magnification


def ensure_odd(v: int) -> int:
    return v if v % 2 == 1 else v + 1


def build_scaled_params(magnification: float, frame_size: tuple[int, int], profile: ROIParams | None = None) -> ScaledParams:
    if profile is None:
        if magnification in MAGNIFICATION_PROFILES:
            profile = MAGNIFICATION_PROFILES[magnification]
        else:
            profile = ROIParams()
    return ScaledParams(profile, magnification, frame_size)


# -----------------------------------------------
# 清晰度评估：拉普拉斯方差
# 值越大 = 边缘越锐利 = 越清晰
# -----------------------------------------------
def compute_sharpness(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


class SimpleTracker:
    def __init__(
        self,
        iou_threshold=0.3,
        max_missing=45,
        roi_pad=35,
        min_track_age=5,
        min_displacement=10.0,
        center_distance_threshold=35.0,
    ):
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.roi_pad = roi_pad
        self.min_track_age = min_track_age
        self.min_displacement = min_displacement
        self.center_distance_threshold = center_distance_threshold
        self.next_id = 0
        self.tracks = {}

    @staticmethod
    def _iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ix = max(0, min(x1 + w1, x2 + w2) - max(x1, x2))
        iy = max(0, min(y1 + h1, y2 + h2) - max(y1, y2))
        inter = ix * iy
        union = w1 * h1 + w2 * h2 - inter
        return inter / union if union > 0 else 0
    
    @staticmethod
    def _center(bbox):
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    def _center_distance(self, b1, b2):
        c1x, c1y = self._center(b1)
        c2x, c2y = self._center(b2)
        dx = c1x - c2x
        dy = c1y - c2y
        return (dx ** 2 + dy ** 2) ** 0.5

    def _crop_roi(self, frame, det):
        frame_h, frame_w = frame.shape[:2]
        x, y, w, h = det
        pad = self.roi_pad
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, frame_w)
        y2 = min(y + h + pad, frame_h)
        return frame[y1:y2, x1:x2]

    def _track_displacement(self, track):
        centers = track["centers"]
        if len(centers) < 2:
            return 0.0

        max_dist = 0.0
        for i in range(len(centers)):
            x1, y1 = centers[i]
            for j in range(i + 1, len(centers)):
                x2, y2 = centers[j]
                dist = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
                if dist > max_dist:
                    max_dist = dist
        return max_dist

    def _should_keep_track(self, track):
        return track["age"] >= self.min_track_age and self._track_displacement(track) >= self.min_displacement

    def update(self, detections, frame):
        matched_track_ids = set()
        matched_det_indices = set()
        new_track_ids = set()

        # 1. 先尝试把 detection 匹配到已有 track
        for tid, track in self.tracks.items():
            best_det_idx = -1
            best_dist = float("inf")
            best_iou = 0.0

            for i, det in enumerate(detections):
                if i in matched_det_indices:
                    continue

                dist = self._center_distance(track["bbox"], det)
                iou = self._iou(track["bbox"], det)

                # 满足“距离近”或“IoU高”才允许匹配
                if dist <= self.center_distance_threshold or iou >= self.iou_threshold:
                    # 优先选距离最近的；如果距离一样，再选 IoU 更高的
                    if dist < best_dist or (abs(dist - best_dist) < 1e-6 and iou > best_iou):
                        best_dist = dist
                        best_iou = iou
                        best_det_idx = i

            if best_det_idx >= 0:
                matched_track_ids.add(tid)
                matched_det_indices.add(best_det_idx)

                det = detections[best_det_idx]
                roi = self._crop_roi(frame, det)
                sharpness = compute_sharpness(roi)
                if sharpness > track["best_sharpness"]:
                    track["best_roi"] = roi.copy()
                    track["best_sharpness"] = sharpness

                track["bbox"] = det
                track["centers"].append(self._center(det))
                track["missing"] = 0
                track["age"] += 1

        # 2. 没匹配上的 detection，新建 track
        for i, det in enumerate(detections):
            if i in matched_det_indices:
                continue

            roi = self._crop_roi(frame, det)
            sharpness = compute_sharpness(roi)

            tid = self.next_id
            self.tracks[tid] = {
                "age": 0,
                "start_bbox": det,
                "bbox": det,
                "centers": [self._center(det)],
                "missing": 0,
                "best_roi": roi.copy(),
                "best_sharpness": sharpness,
            }
            new_track_ids.add(tid)
            self.next_id += 1

        # 3. 只给“旧的且这帧没匹配上”的 track 增加 missing
        finished = []
        ids_to_remove = []

        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids and tid not in new_track_ids:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    track = self.tracks[tid]
                    if self._should_keep_track(track):
                        finished.append(track)
                    ids_to_remove.append(tid)

        for tid in ids_to_remove:
            del self.tracks[tid]

        return finished
    
    def flush(self):
        remaining = [t for t in self.tracks.values() if self._should_keep_track(t)]
        self.tracks.clear()
        return remaining


def contour_touches_frame_edge(cnt, frame_w, frame_h, margin):
    for point in cnt:
        px, py = point[0]
        if px <= margin or py <= margin or px >= frame_w - margin or py >= frame_h - margin:
            return True
    return False


def should_reject_edge_contour(cnt, frame_w, frame_h, area, w, h, params: ScaledParams):
    touches_edge = contour_touches_frame_edge(cnt, frame_w, frame_h, params.edge_margin)
    if not touches_edge:
        return False

    is_large_object = (
        w >= params.large_object_w
        or h >= params.large_object_h
        or area >= params.large_object_area
    )
    if is_large_object:
        return True

    return not params.allow_small_edge_objects


def save_track(track, output_dir, roi_id, predictor, enable_cnn):
    """
    保存一条 track 的最佳 ROI，可选 CNN 推理。
    - enable_cnn=True ：调用 predictor，junk 直接跳过不保存，target 才保存
    - enable_cnn=False：跳过推理，直接保存所有 ROI
    返回 True 表示实际写了文件，False 表示被 CNN 过滤掉了。
    """
    roi_img = track["best_roi"]

    if enable_cnn:
        # predictor.predict() 返回 (pred_label, conf) 元组
        pred_label, conf = predictor.predict(roi_img)

        if pred_label == "junk":
            return False  # junk 不保存

        # target：文件名带置信度，方便后续筛查
        save_path = output_dir / f"roi_{roi_id:05d}_{pred_label}_{conf:.2f}.png"
    else:
        save_path = output_dir / f"roi_{roi_id:05d}.png"

    cv2.imwrite(str(save_path), roi_img)
    return True


# -----------------------------------------------
# 主函数
# -----------------------------------------------
def extract_roi(magnification=1.0, enable_cnn=True):
    base_dir = Path(__file__).resolve().parent.parent.parent
    video_path = base_dir / "data" / "video" / "sample.avi"
    output_dir = base_dir / "data" / "picture_best"
    debug_dir = base_dir / "data" / "debug_best"

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("video path:", video_path)
    print("video exists:", video_path.exists())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: video cannot be opened.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # enable_cnn 参数优先于 profile 里的设置
    profile = MAGNIFICATION_PROFILES.get(magnification, ROIParams())
    profile.enable_cnn = enable_cnn
    params = build_scaled_params(magnification, (frame_w, frame_h), profile)

    print(f"Using magnification: {magnification}x")
    print(f"CNN enabled: {params.enable_cnn}")
    print(f"Frame size: {frame_w}x{frame_h}")
    print(f"Length scale: {params.length_scale:.3f}")
    print(
        "Scaled params:",
        {
            "min_area": params.min_area,
            "max_area": params.max_area,
            "min_w": params.min_w,
            "min_h": params.min_h,
            "max_w": params.max_w,
            "max_h": params.max_h,
            "edge_margin": params.edge_margin,
            "roi_pad": params.roi_pad,
            "min_displacement": round(params.min_displacement, 2),
            "kernels": (params.fill_kernel, params.small_kernel, params.merge_kernel),
        },
    )

    back_sub = create_background_subtractor()
    tracker = SimpleTracker(
        iou_threshold=params.tracker_iou_threshold,
        max_missing=params.tracker_max_missing,
        roi_pad=params.roi_pad,
        min_track_age=params.min_track_age,
        min_displacement=params.min_displacement,
        center_distance_threshold=max(params.min_w, params.min_h) * 1.2,
    )

    # 仅在需要时加载模型，避免不必要的初始化开销
    predictor = ROIPredictor() if params.enable_cnn else None

    frame_id = 0
    roi_id = 0

    fill_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.fill_kernel, params.fill_kernel))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.small_kernel, params.small_kernel))
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (params.merge_kernel, params.merge_kernel))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        frame_h, frame_w = frame.shape[:2]

        fg_mask = subtract_background(back_sub, frame)

        if frame_id <= params.warmup_frames:
            print(f"frame {frame_id} warming up...")
            continue

        fg_filled = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, fill_kernel)
        fg_filled = cv2.morphologyEx(fg_filled, cv2.MORPH_OPEN, small_kernel)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        gray_thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=51,  # 局部区域大小，可调
            C=-10          # 负值=比局部均值亮的才算前景
        )
        gray_thresh = cv2.morphologyEx(gray_thresh, cv2.MORPH_OPEN, small_kernel)

        combined = cv2.bitwise_and(fg_filled, gray_thresh)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, merge_kernel)

        if frame_id <= params.debug_save_until_frame:
            cv2.imwrite(str(debug_dir / f"frame_{frame_id}.png"), frame)
            cv2.imwrite(str(debug_dir / f"fg_mask_{frame_id}.png"), fg_mask)
            cv2.imwrite(str(debug_dir / f"thresh_{frame_id}.png"), combined)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < params.min_area or area > params.max_area:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            if w < params.min_w or h < params.min_h or w > params.max_w or h > params.max_h:
                continue

            aspect_ratio = w / h if h != 0 else 0
            if aspect_ratio < params.min_aspect_ratio or aspect_ratio > params.max_aspect_ratio:
                continue

            if should_reject_edge_contour(cnt, frame_w, frame_h, area, w, h, params):
                continue

            detections.append((x, y, w, h))

        finished_tracks = tracker.update(detections, frame)
        for track in finished_tracks:
            saved = save_track(track, output_dir, roi_id, predictor, params.enable_cnn)
            if saved:
                roi_id += 1

        print(f"frame {frame_id} | detections: {len(detections)} | saved ROI: {roi_id}")

    for track in tracker.flush():
        saved = save_track(track, output_dir, roi_id, predictor, params.enable_cnn)
        if saved:
            roi_id += 1

    cap.release()
    print("Finished!")
    print("CNN enabled:", params.enable_cnn)
    print("Total ROI saved:", roi_id)


if __name__ == "__main__":
    # 开启 CNN（默认）：只保存 target，过滤 junk
    extract_roi(magnification=1.0, enable_cnn=True)

    # 关闭 CNN：保存所有 ROI，不做分类
    # extract_roi(magnification=1.0, enable_cnn=False)