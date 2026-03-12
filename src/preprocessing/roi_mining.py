import cv2
import numpy as np
from pathlib import Path
from preprocessing.background_subtraction import (
    create_background_subtractor,
    subtract_background
)


# -----------------------------------------------
# 清晰度评估：拉普拉斯方差
# 值越大 = 边缘越锐利 = 越清晰
# -----------------------------------------------
def compute_sharpness(roi_bgr):
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


# -----------------------------------------------
# 简单目标追踪器
# 用IoU匹配当前帧检测框和上一帧已有轨迹
# 避免同一只水蚤被重复保存
# -----------------------------------------------
class SimpleTracker:
    def __init__(self, iou_threshold=0.3, max_missing=8):
        """
        iou_threshold: 低于此值认为是新目标
        max_missing:   目标连续消失超过此帧数，则结束该轨迹并保存最优ROI
        """
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.next_id = 0

        # 每条轨迹的结构：
        # {
        #   "bbox": (x,y,w,h),        # 最近一次检测到的bbox
        #   "missing": int,            # 连续未检测到的帧数
        #   "best_roi": np.ndarray,    # 目前最清晰的ROI图像
        #   "best_sharpness": float    # 目前最高清晰度值
        # }
        self.tracks = {}

    @staticmethod
    def _iou(b1, b2):
        """计算两个bbox的IoU，bbox格式为(x,y,w,h)"""
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2
        ix = max(0, min(x1+w1, x2+w2) - max(x1, x2))
        iy = max(0, min(y1+h1, y2+h2) - max(y1, y2))
        inter = ix * iy
        union = w1*h1 + w2*h2 - inter
        return inter / union if union > 0 else 0

    def update(self, detections, frame):
        """
        detections: [(x,y,w,h), ...]  当前帧所有检测框
        frame:      当前帧原图（用于裁剪ROI）
        返回：本帧结束的轨迹列表，每条包含最优ROI
        """
        frame_h, frame_w = frame.shape[:2]

        # --- 步骤1：匹配检测框到已有轨迹 ---
        matched_track_ids = set()
        matched_det_indices = set()

        for tid, track in self.tracks.items():
            best_iou = self.iou_threshold
            best_det_idx = -1
            for i, det in enumerate(detections):
                if i in matched_det_indices:
                    continue
                iou = self._iou(track["bbox"], det)
                if iou > best_iou:
                    best_iou = iou
                    best_det_idx = i

            if best_det_idx >= 0:
                # 匹配成功：更新轨迹
                matched_track_ids.add(tid)
                matched_det_indices.add(best_det_idx)

                x, y, w, h = detections[best_det_idx]
                pad = 15
                x1 = max(x - pad, 0)
                y1 = max(y - pad, 0)
                x2 = min(x + w + pad, frame_w)
                y2 = min(y + h + pad, frame_h)
                roi = frame[y1:y2, x1:x2]

                sharpness = compute_sharpness(roi)
                if sharpness > track["best_sharpness"]:
                    track["best_roi"] = roi.copy()
                    track["best_sharpness"] = sharpness

                track["bbox"] = detections[best_det_idx]
                track["missing"] = 0
                track["age"] += 1

        # --- 步骤2：未匹配的检测框 → 新建轨迹 ---
        for i, det in enumerate(detections):
            if i in matched_det_indices:
                continue
            x, y, w, h = det
            pad = 15
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, frame_w)
            y2 = min(y + h + pad, frame_h)
            roi = frame[y1:y2, x1:x2]
            sharpness = compute_sharpness(roi)

            self.tracks[self.next_id] = {
                "age": 0,  # 累计出现帧数
                "bbox": det,
                "missing": 0,
                "best_roi": roi.copy(),
                "best_sharpness": sharpness,
            }
            self.next_id += 1

        # --- 步骤3：未匹配的轨迹 → missing+1，超限则结束 ---
        finished = []
        ids_to_remove = []
        for tid in list(self.tracks.keys()):
            if tid not in matched_track_ids:
                self.tracks[tid]["missing"] += 1
                if self.tracks[tid]["missing"] > self.max_missing:
                    if self.tracks[tid]["age"] >= 5:   # ← 出现不足5帧的丢弃
                        finished.append(self.tracks[tid])
                    ids_to_remove.append(tid)
        for tid in ids_to_remove:
            del self.tracks[tid]

        return finished

    def flush(self):
        """视频结束时，把所有未结束的轨迹也输出"""
        remaining = [t for t in self.tracks.values() if t["age"] >= 5]
        self.tracks.clear()
        return remaining


# -----------------------------------------------
# 主函数
# -----------------------------------------------
def extract_roi():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    video_path = BASE_DIR / "data" / "video" / "sample.avi"
    output_dir = BASE_DIR / "data" / "picture_best"
    debug_dir = BASE_DIR / "data" / "debug_best"

    output_dir.mkdir(parents=True, exist_ok=True)

    back_sub = create_background_subtractor()
    tracker = SimpleTracker(iou_threshold=0.3, max_missing=8)

    print("video path:", video_path)
    print("video exists:", video_path.exists())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: video cannot be opened.")
        return

    frame_id = 0
    roi_id = 0

    # 形态学核
    fill_kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))  # 填充MOG2空洞
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))    # 去噪
    merge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))  # 融合后再闭运算

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # 1. MOG2背景减除
        fg_mask = subtract_background(back_sub, frame)

        # 预热阶段：让背景模型先收敛，不保存ROI
        if frame_id <= 30:
            print(f"frame {frame_id} warming up...")
            continue

        # 2. 大核闭运算：填充MOG2 mask内部空洞（解决半只水蚤问题）
        fg_filled = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, fill_kernel)
        fg_filled = cv2.morphologyEx(fg_filled, cv2.MORPH_OPEN,  small_kernel)

        # 3. 灰度阈值：提取完整亮目标主体
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        _, gray_thresh = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY)
        gray_thresh = cv2.morphologyEx(gray_thresh, cv2.MORPH_OPEN, small_kernel)

        # 4. 融合：AND运算，既有运动又是亮目标
        combined = cv2.bitwise_and(fg_filled, gray_thresh)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, merge_kernel)

        # 调试：预热后前5帧保存各阶段图
        if frame_id <= 35:
            cv2.imwrite(str(debug_dir / f"frame_{frame_id}.png"), frame)
            cv2.imwrite(str(debug_dir / f"fg_mask_{frame_id}.png"), fg_mask)
            cv2.imwrite(str(debug_dir / f"thresh_{frame_id}.png"), thresh)

        # 5. 找轮廓
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 6. 过滤，得到当前帧的检测框列表
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 700 or area > 20000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            if w < 25 or h < 25 or w > 300 or h > 300:
                continue

            aspect_ratio = w / h if h != 0 else 0
            if aspect_ratio < 0.15 or aspect_ratio > 6.5:
                continue

            margin = 35
            if (x <= margin or y <= margin or
                    x + w >= frame_w - margin or
                    y + h >= frame_h - margin):
                continue

            detections.append((x, y, w, h))

        # 7. 更新追踪器，收集已结束轨迹的最优ROI
        finished_tracks = tracker.update(detections, frame)
        for track in finished_tracks:
            save_path = output_dir / f"roi_{roi_id:05d}.png"
            cv2.imwrite(str(save_path), track["best_roi"])
            roi_id += 1

        print(f"frame {frame_id} | detections: {len(detections)} | saved ROI: {roi_id}")

    # 8. 视频结束，flush剩余轨迹
    for track in tracker.flush():
        save_path = output_dir / f"roi_{roi_id:05d}.png"
        cv2.imwrite(str(save_path), track["best_roi"])
        roi_id += 1

    cap.release()
    print("Finished!")
    print("Total ROI saved:", roi_id)