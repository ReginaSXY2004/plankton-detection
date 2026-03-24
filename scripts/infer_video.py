from pathlib import Path
import csv
import cv2
from ultralytics import YOLO

from microbe_tracker import MicrobeTracker, Detection

# ===== 路径 =====
MODEL_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\yolov8n_microbe_1x\weights\best.pt"
VIDEO_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\video1x\Sample.avi"
OUT_VIDEO = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\microbe_track_custom.avi"
OUT_CSV = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\microbe_track_custom.csv"

CONF = 0.40
IMGSZ = 640
DEVICE = 0


def draw_track(frame, tr):
    x1, y1, x2, y2 = map(int, tr.bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    text = f"id:{tr.track_id} conf:{tr.conf:.2f}"
    cv2.putText(frame, text, (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)


def main():
    model = YOLO(MODEL_PATH)

    tracker = MicrobeTracker(
        max_missing=10,               # 可先试 8~15
        min_hits_to_show=2,           # 至少连续命中2帧才显示
        base_distance_thresh=22.0,    # 基础最大移动距离（像素）
        distance_scale=1.8,           # 按框大小放大允许距离
        max_size_ratio=2.5,           # 前后框大小变化别太夸张
        conf_threshold_for_tracking=0.35,
    )

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
                            x1=x1, y1=y1, x2=x2, y2=y2,
                            conf=float(confs[i]),
                            cls_id=int(clses[i])
                        )
                    )

            tracks = tracker.update(detections)

            # 画框 + 写 CSV
            for tr in tracks:
                draw_track(frame, tr)
                x1, y1, x2, y2 = tr.bbox
                w = x2 - x1
                h = y2 - y1
                csv_writer.writerow([
                    frame_idx, tr.track_id, round(tr.conf, 4),
                    round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2),
                    round(tr.cx, 2), round(tr.cy, 2), round(w, 2), round(h, 2),
                    tr.hits, tr.missed
                ])

            cv2.putText(frame, f"frame:{frame_idx}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)

            writer.write(frame)
            frame_idx += 1

    cap.release()
    writer.release()
    print(f"完成：{OUT_VIDEO}")
    print(f"CSV：{OUT_CSV}")


if __name__ == "__main__":
    main()