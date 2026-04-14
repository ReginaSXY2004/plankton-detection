from pathlib import Path

import cv2
from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "runs" / "yolov8n_microbe_1x" / "weights" / "best.pt"
VIDEO_DIR = BASE_DIR / "data" / "video1x"
OUTPUT_DIR = BASE_DIR / "data" / "yolo_dataset_newwater_pseudo"

SELECTED_VIDEOS = [
    "sample15.avi",
    "sample16.avi",
    "sample17.avi",
]

FRAME_STRIDE = 5

CONF = 0.25
IMGSZ = 800
DEVICE = 0
SAVE_DEBUG_VIS = True
MIN_BOXES_TO_SAVE_FRAME = 1


def should_sample_frame(frame_idx: int) -> bool:
    return frame_idx % FRAME_STRIDE == 0


def clip_box_xyxy(x1: float, y1: float, x2: float, y2: float, frame_w: int, frame_h: int):
    x1 = max(0.0, min(x1, frame_w - 1))
    y1 = max(0.0, min(y1, frame_h - 1))
    x2 = max(0.0, min(x2, frame_w))
    y2 = max(0.0, min(y2, frame_h))
    return x1, y1, x2, y2


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float, frame_w: int, frame_h: int):
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)
    xc = (x1 + bw / 2.0) / frame_w
    yc = (y1 + bh / 2.0) / frame_h
    wn = bw / frame_w
    hn = bh / frame_h
    return xc, yc, wn, hn


def draw_boxes(frame, boxes, color=(0, 255, 0), thickness=2):
    vis = frame.copy()
    for x1, y1, x2, y2, conf in boxes:
        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(vis, p1, p2, color, thickness)
        cv2.putText(
            vis,
            f"{conf:.2f}",
            (p1[0], max(20, p1[1] - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def export_video_pseudolabels(model: YOLO, video_path: Path, images_dir: Path, labels_dir: Path, debug_dir: Path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warning] cannot open video: {video_path}")
        return 0, 0

    saved_frames = 0
    saved_boxes = 0
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if not should_sample_frame(frame_idx):
            continue

        result = model.predict(
            source=frame,
            conf=CONF,
            imgsz=IMGSZ,
            device=DEVICE,
            verbose=False,
        )[0]

        frame_h, frame_w = frame.shape[:2]
        boxes_to_save = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(float, box)
                x1, y1, x2, y2 = clip_box_xyxy(x1, y1, x2, y2, frame_w, frame_h)
                if x2 <= x1 or y2 <= y1:
                    continue
                boxes_to_save.append((x1, y1, x2, y2, float(confs[i])))

        if len(boxes_to_save) < MIN_BOXES_TO_SAVE_FRAME:
            continue

        image_name = f"{video_path.stem}_{frame_idx:06d}.png"
        label_name = f"{video_path.stem}_{frame_idx:06d}.txt"
        image_path = images_dir / image_name
        label_path = labels_dir / label_name

        ok = cv2.imwrite(str(image_path), frame)
        if not ok:
            print(f"[warning] failed to save image: {image_path}")
            continue

        with open(label_path, "w", encoding="utf-8") as f:
            for x1, y1, x2, y2, _ in boxes_to_save:
                xc, yc, wn, hn = xyxy_to_yolo(x1, y1, x2, y2, frame_w, frame_h)
                f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

        if SAVE_DEBUG_VIS:
            vis = draw_boxes(frame, boxes_to_save)
            cv2.imwrite(str(debug_dir / image_name), vis)

        saved_frames += 1
        saved_boxes += len(boxes_to_save)
        print(f"{video_path.name} | frame {frame_idx:06d} | saved_boxes={len(boxes_to_save)}")

    cap.release()
    return saved_frames, saved_boxes


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"model not found: {MODEL_PATH}")

    missing_videos = [name for name in SELECTED_VIDEOS if not (VIDEO_DIR / name).exists()]
    if missing_videos:
        raise FileNotFoundError(f"missing videos: {missing_videos}")

    images_dir = OUTPUT_DIR / "images"
    labels_dir = OUTPUT_DIR / "labels_raw"
    debug_dir = OUTPUT_DIR / "debug_vis"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    if SAVE_DEBUG_VIS:
        debug_dir.mkdir(parents=True, exist_ok=True)

    classes_txt = labels_dir / "classes.txt"
    classes_txt.write_text("microbe\n", encoding="utf-8")

    model = YOLO(str(MODEL_PATH))

    print(f"model: {MODEL_PATH}")
    print(f"video dir: {VIDEO_DIR}")
    print(f"output dir: {OUTPUT_DIR}")
    print(f"selected videos: {SELECTED_VIDEOS}")
    print(f"sampling: every {FRAME_STRIDE} frames")
    print(f"conf: {CONF} | imgsz: {IMGSZ}")

    total_saved_frames = 0
    total_saved_boxes = 0

    for video_name in SELECTED_VIDEOS:
        video_path = VIDEO_DIR / video_name
        print(f"\nprocessing: {video_path.name}")
        saved_frames, saved_boxes = export_video_pseudolabels(
            model,
            video_path,
            images_dir,
            labels_dir,
            debug_dir,
        )
        total_saved_frames += saved_frames
        total_saved_boxes += saved_boxes
        print(
            f"finished {video_path.name} | "
            f"saved_frames={saved_frames} | "
            f"saved_boxes={saved_boxes}"
        )

    print("\npseudo-label dataset finished")
    print(f"total saved frames: {total_saved_frames}")
    print(f"total saved boxes: {total_saved_boxes}")


if __name__ == "__main__":
    main()
