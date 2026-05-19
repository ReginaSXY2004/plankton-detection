"""
运行方式：
cd ~/plankton-detection
source venv_jetson/bin/activate
PYTHONPATH=. python3 scripts/dataset/get_screenshot_dataset.py
"""

from pathlib import Path

import cv2
from ultralytics import YOLO


# =========================
# 路径配置：自动定位项目根目录
# 当前文件位置：
# plankton-detection/scripts/dataset/get_screenshot_dataset.py
# parents[2] = plankton-detection
# =========================
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "runs" / "yolov8n_1x_multiclass_v1" / "weights" / "best.pt"

# 原始视频目录
VIDEO_DIR = PROJECT_ROOT / "data" / "video1x"

# 输出 YOLO 数据集目录
OUTPUT_DIR = PROJECT_ROOT / "data" / "screenshot_1x_sample18-27"


# =========================
# 视频选择配置
# 如果 SELECTED_VIDEOS 为空列表 []，就自动遍历 VIDEO_DIR 下所有视频
# 如果写了文件名，就只处理指定视频
# =========================
SELECTED_VIDEOS = [
    "sample20.avi",
    "sample21.avi",
    "sample22.avi", 
    # "sample23.avi", ......
]

VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv"}


# =========================
# 采样与推理参数
# =========================
FRAME_STRIDE = 20          # 每隔 20 帧截图一次
CONF = 0.25
IMGSZ = 800                # 推理尺寸，太慢可改 640
DEVICE = 0

SAVE_DEBUG_VIS = True      # 是否保存画框检查图
MIN_BOXES_TO_SAVE_FRAME = 1  # 至少检测到几个框才保存该帧


# =========================
# 多分类类别配置
# 必须和使用的训练模型的 data.yaml 一致
# =========================
CLASS_NAMES = {
    0: "daxingzao",
    1: "jianshuizao",
    2: "xiannvchong",
    3: "lunchong",
    4: "xiangbizao",
    5: "weizhi",
    6: "xianchong",
}

# =========================
# debug 可视化颜色
# OpenCV 是 BGR，不是 RGB
# =========================
CLASS_COLORS = {
    0: (255, 80, 80),
    1: (80, 255, 255),
    2: (80, 255, 80),
    3: (80, 80, 255),
    4: (180, 80, 255),
    5: (255, 80, 200),
    6: (255, 180, 80),
}


def should_sample_frame(frame_idx: int) -> bool:
    """判断当前帧是否需要采样。"""
    return frame_idx % FRAME_STRIDE == 0


def clip_box_xyxy(x1, y1, x2, y2, frame_w, frame_h):
    """把检测框限制在图像范围内，防止坐标越界。"""
    x1 = max(0.0, min(float(x1), frame_w - 1))
    y1 = max(0.0, min(float(y1), frame_h - 1))
    x2 = max(0.0, min(float(x2), frame_w))
    y2 = max(0.0, min(float(y2), frame_h))
    return x1, y1, x2, y2


def xyxy_to_yolo(x1, y1, x2, y2, frame_w, frame_h):
    """把 xyxy 像素坐标转换成 YOLO 格式：x_center y_center width height，且归一化到 0-1。"""
    bw = max(0.0, x2 - x1)
    bh = max(0.0, y2 - y1)

    xc = (x1 + bw / 2.0) / frame_w
    yc = (y1 + bh / 2.0) / frame_h
    wn = bw / frame_w
    hn = bh / frame_h

    return xc, yc, wn, hn


def draw_boxes(frame, boxes, thickness=2):
    """
    保存 debug 可视化图。
    boxes 格式：
    (class_id, x1, y1, x2, y2, conf)
    """
    vis = frame.copy()

    for class_id, x1, y1, x2, y2, conf in boxes:
        class_name = CLASS_NAMES.get(class_id, f"class_{class_id}")

        p1 = (int(round(x1)), int(round(y1)))
        p2 = (int(round(x2)), int(round(y2)))

        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        cv2.rectangle(vis, p1, p2, color, thickness)

        text = f"{class_name} {conf:.2f}"
        text_pos = (p1[0], max(20, p1[1] - 6))

        cv2.putText(
            vis,
            text,
            text_pos,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )

    return vis


def get_video_paths(video_dir: Path) -> list[Path]:
    """根据 SELECTED_VIDEOS 获取需要处理的视频列表。"""
    if SELECTED_VIDEOS:
        video_paths = [video_dir / name for name in SELECTED_VIDEOS]

        missing = [str(path) for path in video_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"以下指定视频不存在：\n" + "\n".join(missing))

        return video_paths

    video_paths = [
        path
        for path in sorted(video_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    if not video_paths:
        raise FileNotFoundError(f"没有在视频目录中找到视频文件：{video_dir}")

    return video_paths


def write_dataset_metadata(output_dir: Path, image_paths: list[Path]):
    """写 train.txt、data.yaml、classes.txt。"""
    train_txt = output_dir / "train.txt"

    lines = [
        path.relative_to(output_dir).as_posix()
        for path in sorted(image_paths)
    ]

    train_txt.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )

    names_yaml = "\n".join(
        [f"  {class_id}: {name}" for class_id, name in CLASS_NAMES.items()]
    )

    yaml_text = f"""# YOLOv8 dataset config
# 自动由 get_screenshot_dataset.py 生成

path: ./
train: train.txt

nc: {len(CLASS_NAMES)}
names:
{names_yaml}
"""

    (output_dir / "data.yaml").write_text(yaml_text, encoding="utf-8")

    classes_text = "\n".join(CLASS_NAMES[i] for i in sorted(CLASS_NAMES)) + "\n"
    (output_dir / "classes.txt").write_text(classes_text, encoding="utf-8")


def export_video_pseudolabels(
    model: YOLO,
    video_path: Path,
    images_dir: Path,
    labels_dir: Path,
    debug_dir: Path,
):
    """遍历单个视频，每隔 FRAME_STRIDE 帧截图，并用 YOLO 模型生成多分类伪标签。"""
    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        print(f"[warning] 无法打开视频：{video_path}")
        return [], 0

    saved_image_paths = []
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
            class_ids = result.boxes.cls.cpu().numpy().astype(int)

            for box, conf, class_id in zip(xyxy, confs, class_ids):
                if class_id not in CLASS_NAMES:
                    continue

                x1, y1, x2, y2 = box
                x1, y1, x2, y2 = clip_box_xyxy(
                    x1, y1, x2, y2, frame_w, frame_h
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes_to_save.append(
                    (int(class_id), x1, y1, x2, y2, float(conf))
                )

        if len(boxes_to_save) < MIN_BOXES_TO_SAVE_FRAME:
            continue

        image_name = f"{video_path.stem}_{frame_idx:06d}.png"
        label_name = f"{video_path.stem}_{frame_idx:06d}.txt"

        image_path = images_dir / image_name
        label_path = labels_dir / label_name

        ok = cv2.imwrite(str(image_path), frame)

        if not ok:
            print(f"[warning] 保存图片失败：{image_path}")
            continue

        with open(label_path, "w", encoding="utf-8") as f:
            for class_id, x1, y1, x2, y2, _ in boxes_to_save:
                xc, yc, wn, hn = xyxy_to_yolo(
                    x1, y1, x2, y2, frame_w, frame_h
                )
                f.write(
                    f"{class_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n"
                )

        if SAVE_DEBUG_VIS:
            vis = draw_boxes(frame, boxes_to_save)
            cv2.imwrite(str(debug_dir / image_name), vis)

        saved_image_paths.append(image_path)
        saved_boxes += len(boxes_to_save)

        print(
            f"{video_path.name} | "
            f"frame={frame_idx:06d} | "
            f"saved_boxes={len(boxes_to_save)}"
        )

    cap.release()

    return saved_image_paths, saved_boxes


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"模型不存在：{MODEL_PATH}")

    if not VIDEO_DIR.exists():
        raise FileNotFoundError(f"视频目录不存在：{VIDEO_DIR}")

    images_dir = OUTPUT_DIR / "images" / "train"
    labels_dir = OUTPUT_DIR / "labels" / "train"
    debug_dir = OUTPUT_DIR / "debug_vis"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    if SAVE_DEBUG_VIS:
        debug_dir.mkdir(parents=True, exist_ok=True)

    video_paths = get_video_paths(VIDEO_DIR)
    model = YOLO(str(MODEL_PATH))

    print("========== screenshot dataset export ==========")
    print(f"project root : {PROJECT_ROOT}")
    print(f"model path   : {MODEL_PATH}")
    print(f"video dir    : {VIDEO_DIR}")
    print(f"output dir   : {OUTPUT_DIR}")
    print(f"videos       : {[p.name for p in video_paths]}")
    print(f"frame stride : {FRAME_STRIDE}")
    print(f"conf         : {CONF}")
    print(f"imgsz        : {IMGSZ}")
    print(f"device       : {DEVICE}")
    print("===============================================")

    all_saved_image_paths = []
    total_saved_boxes = 0

    for video_path in video_paths:
        print(f"\n开始处理：{video_path.name}")

        saved_image_paths, saved_boxes = export_video_pseudolabels(
            model=model,
            video_path=video_path,
            images_dir=images_dir,
            labels_dir=labels_dir,
            debug_dir=debug_dir,
        )

        all_saved_image_paths.extend(saved_image_paths)
        total_saved_boxes += saved_boxes

        print(
            f"完成：{video_path.name} | "
            f"saved_frames={len(saved_image_paths)} | "
            f"saved_boxes={saved_boxes}"
        )

    write_dataset_metadata(OUTPUT_DIR, all_saved_image_paths)

    print("\n========== finished ==========")
    print(f"total saved frames : {len(all_saved_image_paths)}")
    print(f"total saved boxes  : {total_saved_boxes}")
    print(f"train.txt          : {OUTPUT_DIR / 'train.txt'}")
    print(f"data.yaml          : {OUTPUT_DIR / 'data.yaml'}")
    print(f"classes.txt        : {OUTPUT_DIR / 'classes.txt'}")
    print("==============================")


if __name__ == "__main__":
    main()