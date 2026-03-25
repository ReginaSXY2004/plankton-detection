from pathlib import Path
import csv
import math

import cv2
import numpy as np
import pandas as pd
import trackpy as tp
from ultralytics import YOLO

from microbe_tracker import Detection  # 只复用 Detection 数据结构，不再用 MicrobeTracker


# ===== 路径 =====
MODEL_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\yolov8n_microbe_1x\weights\best.pt"
VIDEO_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\video0.5x\Sample10.avi"
OUT_VIDEO = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\microbe_track_trackpy.mp4"
OUT_CSV = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\microbe_track_trackpy.csv"

DEVICE = 0

# ===== 当前视频倍率 =====
MAGNIFICATION = 0.5


def get_infer_config(magnification: float):
    """
    前半段检测参数尽量沿用你当前逻辑；
    后半段新增 trackpy 参数。
    """
    config = {
        "conf": 0.25,
        "imgsz": 640,
        "dedup_iou": 0.65,
        "dedup_center": 18,
        "trackpy": dict(
            search_range=22,   # 单帧最大位移
            memory=8,          # 允许消失几帧后接回
            min_track_len=10,  # 至少出现多少帧才显示/写出
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
            "trackpy": dict(
                search_range=16,
                memory=10,
                min_track_len=8,
            )
        }
        return config

    elif magnification == 0.2:
        config = {
            "conf": 0.12,
            "imgsz": 1280,
            "dedup_iou": 0.55,
            "dedup_center": 8,
            "trackpy": dict(
                search_range=10,
                memory=12,
                min_track_len=5,
            )
        }
        return config

    elif magnification == 2.0:
        config = {
            "conf": 0.28,
            "imgsz": 640,
            "dedup_iou": 0.70,
            "dedup_center": 28,
            "trackpy": dict(
                search_range=28,
                memory=6,
                min_track_len=4,
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


def filter_blob_like_detections(
    frame,
    detections,
    circularity_thresh=0.82,
    texture_std_thresh=18.0,
    min_box_size=10
):
    kept = []

    for det in detections:
        bw = det.x2 - det.x1
        bh = det.y2 - det.y1

        # 太小的框先不过滤太狠
        if bw < min_box_size or bh < min_box_size:
            kept.append(det)
            continue

        circularity, texture_std = compute_circularity_from_roi(frame, det)

        if circularity is None:
            kept.append(det)
            continue

        # 很圆 + 纹理很弱，判成光晕/光斑
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


def draw_track(frame, row):
    x1, y1, x2, y2 = map(int, [row["x1"], row["y1"], row["x2"], row["y2"]])
    track_id = int(row["track_id"])
    conf = float(row["conf"])

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"id:{track_id} conf:{conf:.2f}"
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


def collect_detections(model, video_path, conf, imgsz, dedup_iou, dedup_center):
    """
    第一遍：逐帧检测，收集所有候选框。
    Trackpy 更适合这种“先收集，再统一 linking”的批处理方式。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video_path}")

    rows = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
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
            iou_thresh=dedup_iou,
            center_thresh=dedup_center
        )

        detections = filter_blob_like_detections(
            frame,
            detections,
            circularity_thresh=0.82,
            texture_std_thresh=18.0,
            min_box_size=10
        )

        for det in detections:
            rows.append({
                "frame": frame_count,
                "x": det.cx,
                "y": det.cy,
                "conf": det.conf,
                "cls_id": det.cls_id,
                "x1": det.x1,
                "y1": det.y1,
                "x2": det.x2,
                "y2": det.y2,
                "w": det.w,
                "h": det.h,
            })

        if frame_count % 50 == 0:
            print(f"[collect] frame={frame_count}, dets={len(detections)}")

        frame_count += 1

    cap.release()

    df = pd.DataFrame(rows)
    print(f"[collect] total_frames={frame_count}, total_detections={len(df)}")
    return df, frame_count


def run_trackpy(df, search_range, memory, min_track_len):
    """
    用 trackpy 只基于中心点做 linking。
    再把原始 bbox 挂回去。
    """
    if df.empty:
        df["particle"] = []
        return df

    # trackpy 要求按 frame 排序
    df = df.sort_values(["frame", "x", "y"]).reset_index(drop=True)

    linked = tp.link_df(
        df[["frame", "x", "y", "conf", "cls_id", "x1", "y1", "x2", "y2", "w", "h"]].copy(),
        search_range=search_range,
        memory=memory
    )

    # 统计每条轨迹长度
    counts = linked["particle"].value_counts()
    valid_particles = counts[counts >= min_track_len].index
    linked = linked[linked["particle"].isin(valid_particles)].copy()

    linked["track_id"] = linked["particle"].astype(int) + 1
    linked = linked.sort_values(["frame", "track_id"]).reset_index(drop=True)

    print(f"[trackpy] kept_tracks={linked['track_id'].nunique() if not linked.empty else 0}")
    return linked


def render_and_save(video_path, linked_df, total_frames, out_video, out_csv):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_video_path = Path(out_video)
    out_video_path.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(out_video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height)
    )

    csv_path = Path(out_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    grouped = {}
    if not linked_df.empty:
        for frame_idx, group in linked_df.groupby("frame"):
            grouped[int(frame_idx)] = group.to_dict("records")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow([
            "frame_idx", "track_id", "conf",
            "x1", "y1", "x2", "y2", "cx", "cy", "w", "h"
        ])

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rows = grouped.get(frame_idx, [])

            for row in rows:
                draw_track(frame, row)

                cx = (row["x1"] + row["x2"]) / 2.0
                cy = (row["y1"] + row["y2"]) / 2.0
                w = row["x2"] - row["x1"]
                h = row["y2"] - row["y1"]

                csv_writer.writerow([
                    frame_idx,
                    int(row["track_id"]),
                    round(float(row["conf"]), 4),
                    round(float(row["x1"]), 2),
                    round(float(row["y1"]), 2),
                    round(float(row["x2"]), 2),
                    round(float(row["y2"]), 2),
                    round(float(cx), 2),
                    round(float(cy), 2),
                    round(float(w), 2),
                    round(float(h), 2),
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
    print(f"完成：{out_video}")
    print(f"CSV：{out_csv}")


def main():
    cfg = get_infer_config(MAGNIFICATION)

    CONF = cfg["conf"]
    IMGSZ = cfg["imgsz"]
    DEDUP_IOU = cfg["dedup_iou"]
    DEDUP_CENTER = cfg["dedup_center"]

    SEARCH_RANGE = cfg["trackpy"]["search_range"]
    MEMORY = cfg["trackpy"]["memory"]
    MIN_TRACK_LEN = cfg["trackpy"]["min_track_len"]

    print("=" * 60)
    print(f"MAGNIFICATION: {MAGNIFICATION}x")
    print(f"CONF: {CONF}")
    print(f"IMGSZ: {IMGSZ}")
    print(f"DEDUP_IOU: {DEDUP_IOU}")
    print(f"DEDUP_CENTER: {DEDUP_CENTER}")
    print(f"TRACKPY search_range: {SEARCH_RANGE}")
    print(f"TRACKPY memory: {MEMORY}")
    print(f"TRACKPY min_track_len: {MIN_TRACK_LEN}")
    print("=" * 60)

    model = YOLO(MODEL_PATH)

    # 第 1 遍：收集所有 detections
    df, total_frames = collect_detections(
        model=model,
        video_path=VIDEO_PATH,
        conf=CONF,
        imgsz=IMGSZ,
        dedup_iou=DEDUP_IOU,
        dedup_center=DEDUP_CENTER,
    )

    # 第 2 步：trackpy linking
    linked_df = run_trackpy(
        df=df,
        search_range=SEARCH_RANGE,
        memory=MEMORY,
        min_track_len=MIN_TRACK_LEN
    )

    # 第 3 步：渲染输出
    render_and_save(
        video_path=VIDEO_PATH,
        linked_df=linked_df,
        total_frames=total_frames,
        out_video=OUT_VIDEO,
        out_csv=OUT_CSV
    )


if __name__ == "__main__":
    main()