import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from preprocessing.background_subtraction import (
    create_background_subtractor,
    subtract_background,
)
from inference.roi_predictor import ROIPredictor


@dataclass
class DatasetParams:
    warmup_frames: int = 30
    frame_stride: int = 5
    min_boxes_to_save_frame: int = 1

    min_area: int = 700
    max_area: int = 20000

    min_w: int = 25
    min_h: int = 25
    max_w: int = 300
    max_h: int = 300

    min_aspect_ratio: float = 0.15
    max_aspect_ratio: float = 6.5

    fill_kernel: int = 15
    small_kernel: int = 4
    merge_kernel: int = 11

    sharpness_threshold: float = 40.0
    nms_iou_threshold: float = 0.35
    save_debug_vis: bool = True

    # CNN 二分类开关
    enable_cnn: bool = True
    cnn_roi_pad: int = 18
    cnn_conf_threshold: float = 0.80


def compute_sharpness(roi):
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def crop_box_with_pad(frame, x, y, w, h, pad):
    frame_h, frame_w = frame.shape[:2]
    x1 = max(x - pad, 0)
    y1 = max(y - pad, 0)
    x2 = min(x + w + pad, frame_w)
    y2 = min(y + h + pad, frame_h)
    return frame[y1:y2, x1:x2]


def bbox_to_yolo(x, y, w, h, frame_w, frame_h):
    xc = (x + w / 2) / frame_w
    yc = (y + h / 2) / frame_h
    wn = w / frame_w
    hn = h / frame_h
    return xc, yc, wn, hn


def clip_box(x, y, w, h, frame_w, frame_h):
    x = max(0, int(x))
    y = max(0, int(y))
    w = int(min(w, frame_w - x))
    h = int(min(h, frame_h - y))
    return x, y, w, h


def iou_xywh(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter_w = max(0, xb - xa)
    inter_h = max(0, yb - ya)
    inter = inter_w * inter_h

    union = w1 * h1 + w2 * h2 - inter
    if union <= 0:
        return 0.0
    return inter / union


def nms_boxes(boxes, scores, iou_thresh=0.35):
    if not boxes:
        return []

    order = np.argsort(scores)[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)

        remain = []
        for j in order[1:]:
            if iou_xywh(boxes[i], boxes[j]) < iou_thresh:
                remain.append(j)
        order = np.array(remain, dtype=int)

    return [boxes[i] for i in keep]


def draw_boxes(frame, boxes, color=(0, 255, 0), thickness=2):
    vis = frame.copy()
    for x, y, w, h in boxes:
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)
    return vis


def detect(frame, fg_mask, params: DatasetParams, predictor=None):
    fill_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.fill_kernel, params.fill_kernel)
    )
    small_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.small_kernel, params.small_kernel)
    )
    merge_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (params.merge_kernel, params.merge_kernel)
    )

    fg_filled = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, fill_kernel)
    fg_filled = cv2.morphologyEx(fg_filled, cv2.MORPH_OPEN, small_kernel)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    gray_thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        -10,
    )
    gray_thresh = cv2.morphologyEx(gray_thresh, cv2.MORPH_OPEN, small_kernel)

    combined = cv2.bitwise_and(fg_filled, gray_thresh)
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, merge_kernel)

    contours, _ = cv2.findContours(
        combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    frame_h, frame_w = frame.shape[:2]

    boxes = []
    scores = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < params.min_area or area > params.max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        x, y, w, h = clip_box(x, y, w, h, frame_w, frame_h)

        if w <= 1 or h <= 1:
            continue
        if w < params.min_w or h < params.min_h:
            continue
        if w > params.max_w or h > params.max_h:
            continue

        ar = w / h if h > 0 else 0
        if ar < params.min_aspect_ratio or ar > params.max_aspect_ratio:
            continue

        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            continue

        sharpness = compute_sharpness(roi)
        if sharpness < params.sharpness_threshold:
            continue

        # 可选 CNN 二分过滤
        if params.enable_cnn and predictor is not None:
            roi_for_cnn = crop_box_with_pad(frame, x, y, w, h, params.cnn_roi_pad)
            if roi_for_cnn.size == 0:
                continue

            pred_label, conf = predictor.predict(roi_for_cnn)
            if pred_label == "junk" and conf >= params.cnn_conf_threshold:
                continue

        score = float(sharpness) + 0.01 * float(area)
        boxes.append((x, y, w, h))
        scores.append(score)

    boxes = nms_boxes(boxes, scores, iou_thresh=params.nms_iou_threshold)
    return boxes


def build_dataset():
    base_dir = Path(__file__).resolve().parent.parent.parent

    video_dir = base_dir / "data" / "video"
    dataset_dir = base_dir / "data" / "yolo_dataset"

    images_dir = dataset_dir / "images"
    labels_dir = dataset_dir / "labels"
    debug_dir = dataset_dir / "debug_vis"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    params = DatasetParams()
    predictor = ROIPredictor() if params.enable_cnn else None

    if params.save_debug_vis:
        debug_dir.mkdir(parents=True, exist_ok=True)

    classes_txt = labels_dir / "classes.txt"
    with open(classes_txt, "w", encoding="utf-8") as f:
        f.write("microbe\njunk\n")

    videos = sorted(video_dir.glob("*.avi"))
    print("videos:", len(videos))
    print("CNN enabled:", params.enable_cnn)

    total_saved_frames = 0
    total_saved_boxes = 0

    for video_path in videos:
        print(f"\nprocessing: {video_path.name}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"[warning] cannot open video: {video_path}")
            continue

        back_sub = create_background_subtractor()

        frame_id = 0
        saved_frames_this_video = 0
        saved_boxes_this_video = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_id += 1

            if frame_id <= params.warmup_frames:
                continue

            if frame_id % params.frame_stride != 0:
                continue

            fg_mask = subtract_background(back_sub, frame)
            frame_boxes = detect(frame, fg_mask, params, predictor=predictor)

            if len(frame_boxes) < params.min_boxes_to_save_frame:
                continue

            frame_h, frame_w = frame.shape[:2]

            image_name = f"{video_path.stem}_{frame_id:06d}.png"
            label_name = f"{video_path.stem}_{frame_id:06d}.txt"

            image_path = images_dir / image_name
            label_path = labels_dir / label_name

            ok = cv2.imwrite(str(image_path), frame)
            if not ok:
                print(f"[warning] failed to save image: {image_path}")
                continue

            with open(label_path, "w", encoding="utf-8") as f:
                for (x, y, w, h) in frame_boxes:
                    xc, yc, wn, hn = bbox_to_yolo(x, y, w, h, frame_w, frame_h)
                    f.write(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")

            if params.save_debug_vis:
                vis = draw_boxes(frame, frame_boxes)
                debug_path = debug_dir / image_name
                cv2.imwrite(str(debug_path), vis)

            saved_frames_this_video += 1
            saved_boxes_this_video += len(frame_boxes)

            print(
                f"{video_path.name} | frame {frame_id:06d} | "
                f"saved_boxes={len(frame_boxes)}"
            )

        cap.release()

        total_saved_frames += saved_frames_this_video
        total_saved_boxes += saved_boxes_this_video

        print(
            f"finished {video_path.name} | "
            f"saved_frames={saved_frames_this_video} | "
            f"saved_boxes={saved_boxes_this_video}"
        )

    print("\ndataset finished")
    print(f"total saved frames: {total_saved_frames}")
    print(f"total saved boxes: {total_saved_boxes}")


if __name__ == "__main__":
    build_dataset()