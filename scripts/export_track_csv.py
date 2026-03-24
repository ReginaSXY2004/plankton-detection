from pathlib import Path
import csv
from ultralytics import YOLO

# ====== 改这几个路径 ======
MODEL_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\yolov8n_microbe_1x\weights\best.pt"
VIDEO_PATH = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\video1x\Sample.avi"
TRACKER_CFG = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\scripts\bytetrack_microbe.yaml"
OUTPUT_CSV = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs\track_analysis\track_results.csv"
# =========================

CONF = 0.40
IMGSZ = 640
DEVICE = 0


def main():
    model = YOLO(MODEL_PATH)

    output_path = Path(OUTPUT_CSV)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = model.track(
        source=VIDEO_PATH,
        conf=CONF,
        imgsz=IMGSZ,
        device=DEVICE,
        tracker=TRACKER_CFG,
        stream=True,
        persist=True,
        save=False,
        verbose=False,
    )

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_idx",
            "track_id",
            "cls_id",
            "cls_name",
            "conf",
            "x1",
            "y1",
            "x2",
            "y2",
            "cx",
            "cy",
            "w",
            "h",
        ])

        for frame_idx, r in enumerate(results):
            boxes = r.boxes

            # 这一帧没有任何检测
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else []
            clses = boxes.cls.cpu().numpy().astype(int) if boxes.cls is not None else []

            # track 模式下才会有 id；如果某帧没有 id，就记成 -1
            if boxes.id is not None:
                ids = boxes.id.cpu().numpy().astype(int)
            else:
                ids = [-1] * len(xyxy)

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(float, box)
                w = x2 - x1
                h = y2 - y1
                cx = x1 + w / 2
                cy = y1 + h / 2

                cls_id = int(clses[i]) if len(clses) > i else -1
                cls_name = model.names.get(cls_id, str(cls_id)) if isinstance(model.names, dict) else str(cls_id)
                conf = float(confs[i]) if len(confs) > i else -1.0
                track_id = int(ids[i]) if len(ids) > i else -1

                writer.writerow([
                    frame_idx,
                    track_id,
                    cls_id,
                    cls_name,
                    round(conf, 4),
                    round(x1, 2),
                    round(y1, 2),
                    round(x2, 2),
                    round(y2, 2),
                    round(cx, 2),
                    round(cy, 2),
                    round(w, 2),
                    round(h, 2),
                ])

    print(f"导出完成: {output_path}")


if __name__ == "__main__":
    main()