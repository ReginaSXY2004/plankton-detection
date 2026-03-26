from pathlib import Path
import csv
import cv2
from ultralytics import YOLO
import math
import numpy as np
from microbe_tracker import MicrobeTracker, Detection
import time

# ===== 路径 =====
MODEL_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\yolov8n_microbe_1x\weights\best.pt"
VIDEO_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\video1x\Sample8.avi"
OUT_VIDEO = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\microbe_track_custom.avi"
OUT_CSV = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\microbe_track_custom.csv"

# 新增：每个有效 ID 的汇总信息
OUT_CONFIRMED_CSV = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\confirmed_microbes.csv"

# 新增：保存每个 ID 最佳图
BEST_CROP_DIR = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\best_crops"

DEVICE = 0

# ===== 当前视频倍率 =====
MAGNIFICATION = 1


SAVE_VIDEO = False
SAVE_CSV = False

# ===== 实时确认 / 存图参数 =====
# 只对“已经 visible 的轨迹”再加这一层工程判定
BEST_MIN_W = 12                 # 框太小不更新最佳图
BEST_MIN_H = 12
BEST_MIN_CONF = 0.20            # 低于这个置信度不更新最佳图
BEST_MIN_SHARPNESS = 60.0       # 低于这个清晰度不更新最佳图（可后续再调）
SAVE_MIN_SHARPNESS = 80.0       # 最终落盘至少达到这个清晰度
SAVE_MIN_CONF = 0.22            # 最终落盘至少达到这个置信度

# 计数规则：visible track 首次达到这里就计数一次
CONFIRM_MIN_HITS = 6

# 最终写 confirmed_csv / 保存 best crop 的时机：
# 轨迹连续丢失多少帧后，认为结束
FINALIZE_MISSED_THRESH = 8


def get_infer_config(magnification: float):
    config = {
        "conf": 0.22,
        "imgsz": 800,
        "dedup_iou": 0.62,
        "dedup_center": 16,
        "tracker": dict(
            max_missing=18,
            min_hits_to_show=6,
            base_distance_thresh=18.0,
            distance_scale=1.4,
            max_size_ratio=2.0,
            conf_threshold_for_tracking=0.22,
            no_spawn_radius=24.0,
        )
    }

    if magnification == 1.0:
        return config

    elif magnification == 0.5:
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

        if bw < min_box_size or bh < min_box_size:
            kept.append(det)
            continue

        circularity, texture_std = compute_circularity_from_roi(frame, det)

        if circularity is None:
            kept.append(det)
            continue

        if circularity > circularity_thresh and texture_std < texture_std_thresh:
            continue

        kept.append(det)

    return kept


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


def compute_sharpness(roi_bgr: np.ndarray) -> float:
    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def safe_crop(frame: np.ndarray, bbox):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))
    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)
    return frame[y1:y2, x1:x2].copy(), (x1, y1, x2, y2)


def should_consider_best(tr, crop: np.ndarray, sharpness: float) -> bool:
    if crop is None or crop.size == 0:
        return False
    h, w = crop.shape[:2]
    if w < BEST_MIN_W or h < BEST_MIN_H:
        return False
    if tr.conf < BEST_MIN_CONF:
        return False
    if sharpness < BEST_MIN_SHARPNESS:
        return False
    return True


def best_score(sharpness: float, conf: float, area: float) -> float:
    # 清晰度优先，其次置信度和面积
    return sharpness * 1.0 + conf * 120.0 + min(area, 2500.0) * 0.01


def draw_track(frame, tr, display_id=None, is_confirmed=False, counted=False):
    x1, y1, x2, y2 = map(int, tr.bbox)
    color = (0, 255, 0) if is_confirmed else (0, 180, 255)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    flags = []
    if is_confirmed:
        flags.append("ok")
    if counted:
        flags.append("counted")
    flag_text = f" [{'|'.join(flags)}]" if flags else ""

    if display_id is None:
        text = f"tmp conf:{tr.conf:.2f}{flag_text}"
    else:
        text = f"id:{display_id} conf:{tr.conf:.2f}{flag_text}"

    cv2.putText(
        frame,
        text,
        (x1, max(15, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA
    )


def finalize_track_record(tid, rec, confirmed_writer, best_crop_dir: Path):
    if rec["finalized"]:
        return

    rec["finalized"] = True

    # 写 confirmed CSV
    confirmed_writer.writerow([
        tid,
        rec["counted"],
        rec["saved"],
        rec["first_frame"],
        rec["last_frame"],
        rec["best_frame"],
        round(rec["best_conf"], 4),
        round(rec["best_sharpness"], 2),
        rec["best_w"],
        rec["best_h"],
    ])

    # 保存最佳图：只对 counted 且 best 质量过关的目标保存
    if (
        rec["counted"]
        and rec["best_crop"] is not None
        and rec["best_sharpness"] >= SAVE_MIN_SHARPNESS
        and rec["best_conf"] >= SAVE_MIN_CONF
    ):
        out_path = best_crop_dir / f"id_{tid:03d}_frame_{rec['best_frame']:05d}.png"
        cv2.imwrite(str(out_path), rec["best_crop"])
        rec["saved"] = True


def main():
    start_time = time.time()
    frame_count = 0
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

    csv_path = Path(OUT_CSV)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    confirmed_csv_path = Path(OUT_CONFIRMED_CSV)
    confirmed_csv_path.parent.mkdir(parents=True, exist_ok=True)

    best_crop_dir = Path(BEST_CROP_DIR)
    best_crop_dir.mkdir(parents=True, exist_ok=True)

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

    # track_id -> record
    track_records = {}
    realtime_count = 0

    with open(csv_path, "w", newline="", encoding="utf-8") as f_track, \
         open(confirmed_csv_path, "w", newline="", encoding="utf-8") as f_confirmed:

        csv_writer = csv.writer(f_track)
        confirmed_writer = csv.writer(f_confirmed)

        csv_writer.writerow([
            "frame_idx", "track_id", "conf",
            "x1", "y1", "x2", "y2", "cx", "cy", "w", "h", "hits", "missed"
        ])

        confirmed_writer.writerow([
            "track_id", "counted", "saved",
            "first_frame", "last_frame", "best_frame",
            "best_conf", "best_sharpness", "best_w", "best_h"
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

            visible_ids = set()

            # 只对 visible tracks 做显示、逐帧CSV、实时确认逻辑
            for tr in tracks:
                tid = tr.track_id
                visible_ids.add(tid)

                if tid not in track_records:
                    track_records[tid] = {
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "best_frame": -1,
                        "best_conf": 0.0,
                        "best_sharpness": 0.0,
                        "best_w": 0,
                        "best_h": 0,
                        "best_crop": None,
                        "best_score": -1e9,
                        "counted": False,
                        "saved": False,
                        "finalized": False,
                        "display_id": None,
                    }

                rec = track_records[tid]
                rec["last_frame"] = frame_idx

                crop, clipped_bbox = safe_crop(frame, tr.bbox)
                sharpness = compute_sharpness(crop) if crop is not None else 0.0
                area = float((tr.x2 - tr.x1) * (tr.y2 - tr.y1))

                if should_consider_best(tr, crop, sharpness):
                    score = best_score(sharpness, tr.conf, area)
                    if score > rec["best_score"]:
                        rec["best_score"] = score
                        rec["best_frame"] = frame_idx
                        rec["best_conf"] = tr.conf
                        rec["best_sharpness"] = sharpness
                        rec["best_w"] = 0 if crop is None else crop.shape[1]
                        rec["best_h"] = 0 if crop is None else crop.shape[0]
                        rec["best_crop"] = None if crop is None else crop.copy()

                # 实时计数：只计一次
                if (not rec["counted"]) and tr.hits >= CONFIRM_MIN_HITS:
                    realtime_count += 1
                    rec["counted"] = True
                    rec["display_id"] = realtime_count

                draw_track(
                    frame,
                    tr,
                    display_id=rec["display_id"],
                    is_confirmed=rec["counted"],
                    counted=rec["counted"]
                )

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

            # 对 tracker 里已经“长期丢失”的轨迹做 finalize
            for tid, tr in list(tracker.tracks.items()):
                if tid not in track_records:
                    continue
                rec = track_records[tid]
                if rec["finalized"]:
                    continue

                if tr.missed >= FINALIZE_MISSED_THRESH:
                    finalize_track_record(tid, rec, confirmed_writer, best_crop_dir)

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

            cv2.putText(
                frame,
                f"count:{realtime_count}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 200, 255),
                2,
                cv2.LINE_AA
            )

            writer.write(frame)
            frame_idx += 1
            # ===== FPS统计 =====
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = frame_count / elapsed
                print(f"[FPS] {fps:.2f}")

        # 视频结束后，把还没 finalize 的也统一 finalize
        for tid, rec in track_records.items():
            if not rec["finalized"]:
                finalize_track_record(tid, rec, confirmed_writer, best_crop_dir)

        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        print(f"\n平均FPS: {avg_fps:.2f}")

    cap.release()
    writer.release()

    print(f"完成：{OUT_VIDEO}")
    print(f"逐帧CSV：{OUT_CSV}")
    print(f"确认目标CSV：{OUT_CONFIRMED_CSV}")
    print(f"最佳图目录：{BEST_CROP_DIR}")
    print(f"实时计数结果：{realtime_count}")


if __name__ == "__main__":
    main()