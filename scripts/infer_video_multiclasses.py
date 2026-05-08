from pathlib import Path
import csv
import cv2
from ultralytics import YOLO
import math
import numpy as np
from collections import Counter
from microbe_tracker import MicrobeTracker, Detection
import time


# ===== 路径 =====
MODEL_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\yolov8n_1x_multiclass_v1\weights\best.pt"
VIDEO_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\video1x\Sample7.avi"
OUT_VIDEO = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\multiclass_track_custom.mp4"

# confirmed 逐帧 debug（可选）
OUT_DEBUG_CSV = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\confirmed_tracks_debug_multiclass.csv"

# confirmed 汇总
OUT_CONFIRMED_CSV = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\confirmed_microbes_multiclass.csv"

# 每个 confirmed ID 的最佳图
BEST_CROP_DIR = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\best_crops_multiclass"

DEVICE = 0

# ===== 当前视频倍率 =====
MAGNIFICATION = 1

# ===== 类别名 =====
CLASS_NAMES = {
    0: "daxingzao",
    1: "jianshuizao",
    2: "xiannvchong",
    3: "lunchong",
    4: "xiangbizao",
    5: "weizhi",
    6: "xianchong",
}

# 可选：给类别一个固定颜色，便于看视频
CLASS_COLORS = {
    0: (255, 80, 80),     # daxingzao
    1: (80, 255, 255),    # jianshuizao
    2: (80, 255, 80),     # xiannvchong
    3: (80, 80, 255),     # lunchong
    4: (180, 80, 255),    # xiangbizao
    5: (255, 80, 200),    # weizhi
    6: (255, 180, 80),    # xianchong
}

# ===== 输出开关 =====
SAVE_VIDEO = True
SAVE_DEBUG_CSV = False
PRINT_FPS = True
SHOW_CLASS_COUNTS_ON_VIDEO = True

# ===== 最佳图参数 =====
BEST_MIN_W = 12
BEST_MIN_H = 12
BEST_MIN_CONF = 0.18
BEST_MIN_SHARPNESS = 6.0

# ===== 计数 / finalize：按秒控制 =====
CONFIRM_SECONDS = 0.20
FINALIZE_MISSED_SECONDS = 0.30

# tracker 允许目标丢失多久，按倍率给不同默认值
MAX_MISSING_SECONDS_BY_MAG = {
    1.0: 0.60,
    0.5: 0.70,
    0.2: 0.50,
    2.0: 0.35,
}

# lost track 在多少秒内允许 reconnect
RECONNECT_SECONDS = 0.5

# ===== 类别投票 =====
# 当轨迹被确认后，用累计投票最多的类别作为该轨迹最终类别
MIN_CLASS_VOTES_TO_LOCK = 3


def get_infer_config(magnification: float):
    config = {
        "conf": 0.35,
        "imgsz": 800,
        "dedup_iou": 0.62,
        "dedup_center": 16,
        "tracker": dict(
            min_hits_to_show=6,
            base_distance_thresh=18.0,
            distance_scale=1.4,
            max_size_ratio=2.0,
            conf_threshold_for_tracking=0.22,
            no_spawn_radius=24.0,
            debug_print=False,
        )
    }

    if magnification == 1.0:
        return config

    elif magnification == 0.5:
        return {
            "conf": 0.35,
            "imgsz": 800,
            "dedup_iou": 0.60,
            "dedup_center": 12,
            "tracker": dict(
                min_hits_to_show=6,
                base_distance_thresh=16.0,
                distance_scale=1.5,
                max_size_ratio=2.0,
                conf_threshold_for_tracking=0.18,
                no_spawn_radius=20.0,
                debug_print=False,
            )
        }

    elif magnification == 0.2:
        return {
            "conf": 0.40,
            "imgsz": 1280,
            "dedup_iou": 0.55,
            "dedup_center": 8,
            "tracker": dict(
                min_hits_to_show=4,
                base_distance_thresh=8.0,
                distance_scale=1.2,
                max_size_ratio=2.0,
                conf_threshold_for_tracking=0.12,
                no_spawn_radius=16.0,
                debug_print=False,
            )
        }

    elif magnification == 2.0:
        return {
            "conf": 0.40,
            "imgsz": 640,
            "dedup_iou": 0.70,
            "dedup_center": 28,
            "tracker": dict(
                min_hits_to_show=3,
                base_distance_thresh=28.0,
                distance_scale=1.5,
                max_size_ratio=2.2,
                conf_threshold_for_tracking=0.28,
                no_spawn_radius=50.0,
                debug_print=False,
            )
        }

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

    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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

def point_in_box(cx, cy, box):
    x1, y1, x2, y2 = box
    return x1 <= cx <= x2 and y1 <= cy <= y2

def is_duplicate_track_candidate(new_tr, old_tr) -> bool:
    """
    判断两个 track 是否很可能是同一个微生物。
    用于 counted 前去重，避免重复计数。
    """
    new_box = new_tr.bbox
    old_box = old_tr.bbox

    iou = box_iou_xyxy(new_box, old_box)
    center_dist = ((new_tr.cx - old_tr.cx) ** 2 + (new_tr.cy - old_tr.cy) ** 2) ** 0.5

    mutual_center_inside = (
        point_in_box(new_tr.cx, new_tr.cy, old_box)
        and point_in_box(old_tr.cx, old_tr.cy, new_box)
    )

    dynamic_center_thresh = 0.35 * max(
        new_tr.w, new_tr.h, old_tr.w, old_tr.h
    )

    similar_size = (
        max(new_tr.w, old_tr.w) / max(min(new_tr.w, old_tr.w), 1e-6) < 1.8
        and max(new_tr.h, old_tr.h) / max(min(new_tr.h, old_tr.h), 1e-6) < 1.8
    )

    return (
        similar_size
        and (
            iou > 0.35
            or mutual_center_inside
            or center_dist < dynamic_center_thresh
        )
    )


def deduplicate_detections(detections, iou_thresh=0.65, center_thresh=18):
    detections = sorted(detections, key=lambda d: d.conf, reverse=True)
    kept = []

    for det in detections:
        keep = True
        for k in kept:
            iou = box_iou_xyxy((det.x1, det.y1, det.x2, det.y2), (k.x1, k.y1, k.x2, k.y2))
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
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

# 这一帧的这个目标，有没有资格参加“最佳截图评选”
def check_best_candidate(tr, crop: np.ndarray, sharpness: float):
    if crop is None or crop.size == 0:
        return False, "empty_crop"

    h, w = crop.shape[:2]

    if w < BEST_MIN_W or h < BEST_MIN_H:
        return False, "too_small"

    if tr.conf < BEST_MIN_CONF:
        return False, "low_conf"

    if sharpness < BEST_MIN_SHARPNESS:
        return False, "low_sharpness"

    return True, "ok"


def best_score(sharpness: float, conf: float, area: float) -> float:
    return sharpness * 1.0 + conf * 120.0 + min(area, 2500.0) * 0.01

def seconds_to_frames(seconds: float, fps: float, min_frames: int = 1) -> int:
    return max(min_frames, int(round(seconds * fps)))

def majority_class_from_votes(class_votes: Counter):
    if not class_votes:
        return None
    cls_id, votes = class_votes.most_common(1)[0]
    if votes < MIN_CLASS_VOTES_TO_LOCK:
        return None
    return cls_id


def draw_confirmed_track(frame, tr, rec):
    x1, y1, x2, y2 = map(int, tr.bbox)
    cls_id = rec["final_cls_id"] if rec["final_cls_id"] is not None else rec["last_cls_id"]
    cls_name = CLASS_NAMES.get(cls_id, f"cls{cls_id}") if cls_id is not None else "unknown"
    color = CLASS_COLORS.get(cls_id, (0, 255, 0))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    text = f"id:{rec['display_id']} {cls_name} {tr.conf:.2f}"
    cv2.putText(
        frame,
        text,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA
    )


def maybe_save_best_crop(rec, best_crop_dir: Path):
    rec["save_fail_reason"] = ""
    if not rec["counted"]:
        rec["save_fail_reason"] = "not_counted"
        return False
    if rec["best_crop"] is None:
        rec["save_fail_reason"] = rec.get("last_best_update_status") or "no_valid_best_crop"
        return False

    show_id = rec["display_id"] if rec["display_id"] is not None else rec["track_id"]
    cls_name = CLASS_NAMES.get(rec["final_cls_id"], "unknown")
    out_path = best_crop_dir / (
        f"{cls_name}_showid_{show_id:03d}_track_{rec['track_id']:03d}_frame_{rec['best_frame']:05d}.png"
    )
    ok = cv2.imwrite(str(out_path), rec["best_crop"])
    rec["save_fail_reason"] = "saved" if ok else "imwrite_failed"
    return ok


def finalize_track_record(rec, confirmed_writer, best_crop_dir: Path):
    if rec["finalized"]:
        return

    rec["finalized"] = True
    rec["saved"] = maybe_save_best_crop(rec, best_crop_dir)

    confirmed_writer.writerow([
        rec["display_id"],
        rec["track_id"],
        rec["counted"],
        rec["saved"],
        rec["final_cls_id"],
        CLASS_NAMES.get(rec["final_cls_id"], "unknown") if rec["final_cls_id"] is not None else "unknown",
        rec["first_frame"],
        rec["last_frame"],
        rec["best_frame"],
        round(rec["best_conf"], 4),
        round(rec["best_sharpness"], 2),
        rec["best_w"],
        rec["best_h"],
        dict(rec["class_votes"]),
        rec["save_fail_reason"],
        rec["last_best_update_status"],
    ])


def draw_class_count_panel(frame, class_counts, total_count):
    x0, y0 = 10, 80
    line_h = 24

    cv2.putText(
        frame, f"total:{total_count}", (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2, cv2.LINE_AA
    )

    sorted_items = sorted(class_counts.items(), key=lambda kv: kv[0])
    for i, (cls_id, cnt) in enumerate(sorted_items, start=1):
        cls_name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.putText(
            frame,
            f"{cls_name}:{cnt}",
            (x0, y0 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )


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
    print(f"MODEL_PATH: {MODEL_PATH}")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    out_video_path = Path(OUT_VIDEO)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    confirmed_csv_path = Path(OUT_CONFIRMED_CSV)
    confirmed_csv_path.parent.mkdir(parents=True, exist_ok=True)

    best_crop_dir = Path(BEST_CROP_DIR)
    best_crop_dir.mkdir(parents=True, exist_ok=True)

    debug_csv_path = Path(OUT_DEBUG_CSV)
    if SAVE_DEBUG_CSV:
        debug_csv_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {VIDEO_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0

    confirm_min_hits = seconds_to_frames(
    CONFIRM_SECONDS,
    src_fps,
    min_frames=3
    )

    finalize_missed_thresh = seconds_to_frames(
        FINALIZE_MISSED_SECONDS,
        src_fps,
        min_frames=3
    )

    max_missing_seconds = MAX_MISSING_SECONDS_BY_MAG.get(float(MAGNIFICATION), 0.60)
    max_missing_frames = seconds_to_frames(
        max_missing_seconds,
        src_fps,
        min_frames=3
    )

    reconnect_max_missing = seconds_to_frames(
        RECONNECT_SECONDS,
        src_fps,
        min_frames=2
    )

    cfg["tracker"]["max_missing"] = max_missing_frames
    cfg["tracker"]["reconnect_max_missing"] = reconnect_max_missing

    print(f"src_fps: {src_fps}")
    print(f"confirm_min_hits: {confirm_min_hits} frames ({CONFIRM_SECONDS}s)")
    print(f"finalize_missed_thresh: {finalize_missed_thresh} frames ({FINALIZE_MISSED_SECONDS}s)")
    print(f"max_missing_frames: {max_missing_frames} frames ({max_missing_seconds}s)")
    print(f"reconnect_max_missing: {reconnect_max_missing} frames ({RECONNECT_SECONDS}s)")

    tracker = MicrobeTracker(**cfg["tracker"])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if SAVE_VIDEO:
        writer = cv2.VideoWriter(
            str(out_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            src_fps,
            (width, height)
        )

    # internal track_id -> record
    track_records = {}
    counted_tracks = {}
    realtime_count = 0
    class_counts = Counter()

    f_debug = None
    debug_writer = None

    try:
        if SAVE_DEBUG_CSV:
            f_debug = open(debug_csv_path, "w", newline="", encoding="utf-8")
            debug_writer = csv.writer(f_debug)
            debug_writer.writerow([
                "frame_idx", "display_id", "track_id", "cls_id", "cls_name", "conf",
                "x1", "y1", "x2", "y2", "cx", "cy", "w", "h", "hits", "missed"
            ])

        with open(confirmed_csv_path, "w", newline="", encoding="utf-8") as f_confirmed:
            confirmed_writer = csv.writer(f_confirmed)
            confirmed_writer.writerow([
                "display_id", "track_id", "counted", "saved",
                "final_cls_id", "final_cls_name",
                "first_frame", "last_frame", "best_frame",
                "best_conf", "best_sharpness", "best_w", "best_h", "class_votes",
                "save_fail_reason", "last_best_update_status"
            ])

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Yolo 检测
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

                # 去重，过滤圆形亮斑
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
                tracks_to_draw = []
                # 清理已经不在 tracker 里的 counted track，避免历史旧目标影响新目标计数
                active_track_ids = set(tracker.tracks.keys())
                for old_tid in list(counted_tracks.keys()):
                    if old_tid not in active_track_ids:
                        del counted_tracks[old_tid]

                for tr in tracks:
                    tid = tr.track_id

                    if tid not in track_records:
                        track_records[tid] = {
                            
                            "track_id": tid,
                            "display_id": None,
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
                            "is_duplicate": False,
                            "duplicate_of": None,
                            "class_votes": Counter(),
                            "final_cls_id": None,
                            "last_cls_id": None,
                            "save_fail_reason": "",
                            "last_best_update_status": "",
                        }

                    rec = track_records[tid]
                    rec["last_frame"] = frame_idx
                    rec["last_cls_id"] = tr.cls_id
                    rec["class_votes"][tr.cls_id] += 1

                    # 更新当前投票主类
                    rec["final_cls_id"] = majority_class_from_votes(rec["class_votes"])

                    # 裁当前帧的原图区域，还没画框
                    crop, _ = safe_crop(frame, tr.bbox)
                    sharpness = compute_sharpness(crop) if crop is not None else 0.0
                    area = float((tr.x2 - tr.x1) * (tr.y2 - tr.y1))

                    # 判断截图有没有资格竞争 best crop
                    ok_best, reason = check_best_candidate(tr, crop, sharpness)

                    if not ok_best:
                        rec["last_best_update_status"] = reason
                    else:
                        # 如果合格，就和历史 best 比分
                        score = best_score(sharpness, tr.conf, area)
                        if score > rec["best_score"]:
                            rec["best_score"] = score
                            rec["best_frame"] = frame_idx
                            rec["best_conf"] = tr.conf
                            rec["best_sharpness"] = sharpness
                            rec["best_w"] = 0 if crop is None else crop.shape[1]
                            rec["best_h"] = 0 if crop is None else crop.shape[0]
                            rec["best_crop"] = None if crop is None else crop.copy() # best crop 每个 track 只保留目前最好的那一张
                            rec["last_best_update_status"] = "accepted_as_best"

                    # track 达到条件之后才counted
                    if (not rec["counted"]) and (not rec["is_duplicate"]) and tr.hits >= confirm_min_hits:
                        duplicate_of = None

                        for old_tid, old_tr in counted_tracks.items():
                            if is_duplicate_track_candidate(tr, old_tr):
                                duplicate_of = old_tid
                                break

                        if duplicate_of is not None:
                            rec["is_duplicate"] = True
                            rec["duplicate_of"] = duplicate_of
                            rec["counted"] = False
                            rec["display_id"] = track_records[duplicate_of]["display_id"]
                        else:
                            realtime_count += 1
                            rec["counted"] = True
                            rec["display_id"] = realtime_count

                            locked_cls = rec["final_cls_id"] if rec["final_cls_id"] is not None else tr.cls_id
                            class_counts[locked_cls] += 1

                            counted_tracks[tid] = tr

                    if rec["counted"] and (not rec["is_duplicate"]):
                        tracks_to_draw.append((tr, rec))

                        if SAVE_DEBUG_CSV and debug_writer is not None:
                            x1, y1, x2, y2 = tr.bbox
                            w = x2 - x1
                            h = y2 - y1
                            cls_id = rec["final_cls_id"] if rec["final_cls_id"] is not None else tr.cls_id
                            debug_writer.writerow([
                                frame_idx,
                                rec["display_id"],
                                tr.track_id,
                                cls_id,
                                CLASS_NAMES.get(cls_id, "unknown"),
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

                # 统一画框
                for tr, rec in tracks_to_draw:
                    draw_confirmed_track(frame, tr, rec)

                for tid, tr in list(tracker.tracks.items()):
                    rec = track_records.get(tid)
                    if rec is None or rec["finalized"]:
                        continue
                    if tr.missed >= finalize_missed_thresh and rec["counted"]:
                        finalize_track_record(rec, confirmed_writer, best_crop_dir)

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

                if SHOW_CLASS_COUNTS_ON_VIDEO:
                    draw_class_count_panel(frame, class_counts, realtime_count)

                if writer is not None:
                    writer.write(frame)

                frame_idx += 1
                frame_count += 1

                if PRINT_FPS and frame_count % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_now = frame_count / max(elapsed, 1e-6)
                    print(f"[FPS] {fps_now:.2f}")

            for rec in track_records.values():
                if rec["counted"] and (not rec["finalized"]):
                    finalize_track_record(rec, confirmed_writer, best_crop_dir)

            total_time = time.time() - start_time
            avg_fps = frame_count / max(total_time, 1e-6)
            print(f"\n平均FPS: {avg_fps:.2f}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if f_debug is not None:
            f_debug.close()

    if SAVE_VIDEO:
        print(f"完成视频：{OUT_VIDEO}")
    else:
        print("完成视频：未保存（SAVE_VIDEO=False）")

    if SAVE_DEBUG_CSV:
        print(f"confirmed 逐帧 debug CSV：{OUT_DEBUG_CSV}")
    else:
        print("confirmed 逐帧 debug CSV：未保存（SAVE_DEBUG_CSV=False）")

    print(f"confirmed 汇总 CSV：{OUT_CONFIRMED_CSV}")
    print(f"最佳图目录：{BEST_CROP_DIR}")
    print(f"实时总计数：{realtime_count}")

    print("分类别计数：")
    for cls_id in sorted(class_counts.keys()):
        print(f"  {CLASS_NAMES.get(cls_id, cls_id)}: {class_counts[cls_id]}")


if __name__ == "__main__":
    main()