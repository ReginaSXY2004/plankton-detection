from pathlib import Path
import csv
import cv2
from ultralytics import YOLO
import math
import numpy as np
from microbe_tracker import MicrobeTracker, Detection

# ===== 路径 =====
MODEL_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\yolov8n_microbe_1x\weights\best.pt"
VIDEO_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\video0.5x\Sample10.avi"
OUT_VIDEO = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\microbe_track_custom.avi"
OUT_CSV = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\microbe_track_custom.csv"

DEVICE = 0

# ===== 当前视频倍率 =====
# 可选：1.0 / 0.5 / 0.2 / 2.0
MAGNIFICATION = 0.5


def get_infer_config(magnification: float):
    """
    方法B：不缩放图像，直接按倍率调整 YOLO 和 tracker 参数
    这些参数是经验版，不是绝对真理，先拿来测试很合适。
    """

    # 默认按 1x
    config = {
        "conf": 0.25,
        "imgsz": 640,
        "dedup_iou": 0.65,
        "dedup_center": 18,
        "tracker": dict(
            max_missing=14,
            min_hits_to_show=10,
            base_distance_thresh=20.0,
            distance_scale=1.4,
            max_size_ratio=2.0,
            conf_threshold_for_tracking=0.25,
            no_spawn_radius=35.0,
        )
    }

    if magnification == 1.0:
        return config

    elif magnification == 0.5:
        # 目标更小、更糊，所以：
        # - 降低检测阈值，提升召回
        # - 提高 imgsz，帮助小目标
        # - tracker 空间阈值整体收缩
        config = {
            "conf": 0.18,
            "imgsz": 960,
            "dedup_iou": 0.60,
            "dedup_center": 12,
            "tracker": dict(
                max_missing=20,
                min_hits_to_show=6,
                base_distance_thresh=16.0,
                distance_scale=1.5,
                max_size_ratio=2.0,
                conf_threshold_for_tracking=0.18,
                no_spawn_radius=20.0,
            )
        }
        return config

    elif magnification == 0.2:
        # 更极端的小目标场景，先给更保守的召回导向参数
        config = {
            "conf": 0.12,
            "imgsz": 1280,
            "dedup_iou": 0.55,
            "dedup_center": 8,
            "tracker": dict(
                max_missing=14,
                min_hits_to_show=4,
                base_distance_thresh=8.0,
                distance_scale=1.2,
                max_size_ratio=2.0,
                conf_threshold_for_tracking=0.12,
                no_spawn_radius=16.0,
            )
        }
        return config

    elif magnification == 2.0:
        # 目标更大，所以可以更严格一点
        config = {
            "conf": 0.28,
            "imgsz": 640,
            "dedup_iou": 0.70,
            "dedup_center": 28,
            "tracker": dict(
                max_missing=8,
                min_hits_to_show=3,
                base_distance_thresh=28.0,
                distance_scale=1.5,
                max_size_ratio=2.2,
                conf_threshold_for_tracking=0.28,
                no_spawn_radius=50.0,
            )
        }
        return config

    else:
        print(f"[warning] 未定义倍率 {magnification}x，自动按 1.0x 参数运行。")
        return config

def compute_circularity_from_roi(frame, det):
    x1, y1, x2, y2 = map(int, [det.x1, det.y1, det.x2, det.y2])

    h, w = frame.shape[:2]
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = frame[y1:y2, x1:x2]
    if roi.size == 0:
        return None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 用 Otsu 自动阈值，尽量抓住亮斑主体
    _, binary = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, float(np.std(gray))

    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter <= 1e-6:
        return None, float(np.std(gray))

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    texture_std = float(np.std(gray))
    return circularity, texture_std

def filter_blob_like_detections(frame, detections,
                                circularity_thresh=0.82,
                                texture_std_thresh=18.0,
                                min_box_size=10):
    kept = []

    for det in detections:
        bw = det.x2 - det.x1
        bh = det.y2 - det.y1

        # 太小的框先不过滤太狠，避免误杀小微生物
        if bw < min_box_size or bh < min_box_size:
            kept.append(det)
            continue

        circularity, texture_std = compute_circularity_from_roi(frame, det)

        # 算不出来就先保留
        if circularity is None:
            kept.append(det)
            continue

        # 很圆 + 纹理很弱，判成光斑
        if circularity > circularity_thresh and texture_std < texture_std_thresh:
            continue

        kept.append(det)

    return kept

def draw_track(frame, tr):
    x1, y1, x2, y2 = map(int, tr.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"id:{tr.track_id} conf:{tr.conf:.2f}"
    cv2.putText(
        frame,
        text,
        (x1, max(15, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-6
    return inter / union


def deduplicate_detections(detections, iou_thresh=0.65, center_thresh=18):
    """
    同一帧里先去掉很近的重复框，避免同一目标喂给 tracker 两次
    """
    detections = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept = []

    for det in detections:
        keep = True
        for k in kept:
            iou = box_iou_xyxy(
                (det.x1, det.y1, det.x2, det.y2),
                (k.x1, k.y1, k.x2, k.y2)
            )
            center_dist = ((det.cx - k.cx) ** 2 + (det.cy - k.cy) ** 2) ** 0.5

            if iou > iou_thresh or center_dist < center_thresh:
                keep = False
                break

        if keep:
            kept.append(det)

    return kept


def main():
    cfg = get_infer_config(MAGNIFICATION)

    CONF = cfg["conf"]
    IMGSZ = cfg["imgsz"]
    DEDUP_IOU = cfg["dedup_iou"]
    DEDUP_CENTER = cfg["dedup_center"]

    print("=" * 60)
    print(f"MAGNIFICATION: {MAGNIFICATION}x")
    print(f"CONF: {CONF}")
    print(f"IMGSZ: {IMGSZ}")
    print(f"DEDUP_IOU: {DEDUP_IOU}")
    print(f"DEDUP_CENTER: {DEDUP_CENTER}")
    print(f"TRACKER_CONFIG: {cfg['tracker']}")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    tracker = MicrobeTracker(**cfg["tracker"])

    out_video_path = Path(OUT_VIDEO)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {VIDEO_PATH}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    csv_path = Path(OUT_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "frame_idx", "track_id", "conf",
            "x1", "y1", "x2", "y2", "cx", "cy", "w", "h", "hits", "missed"
        ])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model.predict(
                source=frame,
                conf=CONF,
                imgsz=IMGSZ,
                device=DEVICE,
                verbose=False
            )[0]

            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                xyxy = result.boxes.xyxy.cpu().numpy()
                confs = result.boxes.conf.cpu().numpy()
                clses = result.boxes.cls.cpu().numpy().astype(int)

                for i, box in enumerate(xyxy):
                    x1, y1, x2, y2 = map(float, box)
                    detections.append(
                        Detection(
                            x1=x1,
                            y1=y1,
                            x2=x2,
                            y2=y2,
                            conf=float(confs[i]),
                            cls_id=int(clses[i])
                        )
                    )

            detections = deduplicate_detections(
                detections,
                iou_thresh=DEDUP_IOU,
                center_thresh=DEDUP_CENTER
            )

            detections = filter_blob_like_detections(
                frame,
                detections,
                circularity_thresh=0.82,
                texture_std_thresh=18.0,
                min_box_size=10
            )

            tracks = tracker.update(detections)

            for tr in tracks:
                draw_track(frame, tr)
                x1, y1, x2, y2 = tr.bbox
                w = x2 - x1
                h = y2 - y1
                csv_writer.writerow([
                    frame_idx,
                    tr.track_id,
                    round(tr.conf, 4),
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                    round(tr.cx, 2),
                    round(tr.cy, 2),
                    round(w, 2),
                    round(h, 2),
                    tr.hits,
                    tr.missed
                ])

            cv2.putText(
                frame,
                f"frame:{frame_idx}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"完成：{OUT_VIDEO}")
    print(f"CSV：{OUT_CSV}")


if __name__ == "__main__":
    main()