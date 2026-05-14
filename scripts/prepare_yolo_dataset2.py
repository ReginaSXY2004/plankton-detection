"""
YOLO 数据集检查与 train.txt / val.txt 生成脚本

适用数据结构：

data/yolo_dataset1x_merged_classification/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── data.yaml
├── train.txt
└── val.txt

功能：
1. 检查 images/train 与 labels/train 是否一一对应
2. 检查 images/val 与 labels/val 是否一一对应
3. 检查 YOLO label 格式是否正确
4. 生成 train.txt / val.txt
5. 输出 train / val 类别统计

注意：
- 本脚本不移动图片或标签
- 本脚本不重新划分 train / val
- train.txt / val.txt 中写入相对于 DATASET_ROOT 的图片路径
"""

from pathlib import Path
from collections import Counter


# scripts/prepare_yolo_dataset2.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "yolo_dataset1x_merged_classification"

IMAGES_TRAIN_DIR = DATASET_ROOT / "images" / "train"
IMAGES_VAL_DIR = DATASET_ROOT / "images" / "val"

LABELS_TRAIN_DIR = DATASET_ROOT / "labels" / "train"
LABELS_VAL_DIR = DATASET_ROOT / "labels" / "val"

TRAIN_TXT = DATASET_ROOT / "train.txt"
VAL_TXT = DATASET_ROOT / "val.txt"

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

CLASS_NAMES = {
    0: "daxingzao",
    1: "jianshuizao",
    2: "xiannvchong",
    3: "lunchong",
    4: "xiangbizao",
    5: "weizhi",
    6: "xianchong",
}


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_SUFFIXES


def get_label_path_from_image(img_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{img_path.stem}.txt"


def parse_label_file(label_path: Path):
    """
    返回：
    - box_count: 该图总框数
    - class_counter: 该图各类框数 Counter

    允许空 txt，表示负样本 / 空图。
    """
    class_counter = Counter()
    box_count = 0

    if not label_path.exists():
        raise FileNotFoundError(f"标签不存在: {label_path}")

    lines = label_path.read_text(encoding="utf-8").splitlines()

    for line_no, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue

        parts = raw.split()
        if len(parts) != 5:
            raise ValueError(f"{label_path} 第 {line_no} 行格式错误: {raw}")

        try:
            cls_id = int(parts[0])
            nums = list(map(float, parts[1:]))
        except Exception as e:
            raise ValueError(f"{label_path} 第 {line_no} 行解析失败: {raw}") from e

        if cls_id not in CLASS_NAMES:
            raise ValueError(f"{label_path} 第 {line_no} 行类别越界: {cls_id}")

        if not all(0.0 <= x <= 1.0 for x in nums):
            raise ValueError(f"{label_path} 第 {line_no} 行坐标疑似未归一化: {raw}")

        class_counter[cls_id] += 1
        box_count += 1

    return box_count, class_counter


def collect_split(split_name: str, images_dir: Path, labels_dir: Path):
    """
    收集一个 split 的样本。

    返回 list[dict]：
    {
        "img_path": Path,
        "label_path": Path,
        "rel_img_path": str,
        "box_count": int,
        "class_counter": Counter,
        "is_empty": bool,
    }
    """
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"数据集根目录不存在: {DATASET_ROOT}")

    if not images_dir.exists():
        raise FileNotFoundError(f"{split_name} 图片目录不存在: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"{split_name} 标签目录不存在: {labels_dir}")

    image_files = sorted([p for p in images_dir.iterdir() if is_image_file(p)])

    if not image_files:
        raise RuntimeError(f"{split_name} 没有在 {images_dir} 找到图片")

    samples = []
    missing_labels = []

    for img_path in image_files:
        label_path = get_label_path_from_image(img_path, labels_dir)

        if not label_path.exists():
            missing_labels.append(label_path)
            continue

        box_count, class_counter = parse_label_file(label_path)

        rel_img_path = img_path.relative_to(DATASET_ROOT).as_posix()

        samples.append({
            "img_path": img_path,
            "label_path": label_path,
            "rel_img_path": rel_img_path,
            "box_count": box_count,
            "class_counter": class_counter,
            "is_empty": (box_count == 0),
        })

    if missing_labels:
        preview = "\n".join(str(p) for p in missing_labels[:10])
        raise RuntimeError(
            f"{split_name} 发现 {len(missing_labels)} 张图片缺少标签文件，前 10 个如下:\n{preview}"
        )

    return samples


def summarize_split(samples, title: str):
    image_count = len(samples)
    empty_count = sum(1 for s in samples if s["is_empty"])
    total_boxes = sum(s["box_count"] for s in samples)

    class_counter = Counter()
    for s in samples:
        class_counter.update(s["class_counter"])

    print(f"\n========== {title} 统计 ==========")
    print(f"图片数: {image_count}")
    print(f"空图数: {empty_count}")
    print(f"总框数: {total_boxes}")
    print("每类框数:")

    for cls_id in sorted(CLASS_NAMES.keys()):
        print(f"  class {cls_id}: {CLASS_NAMES[cls_id]:<12} -> {class_counter.get(cls_id, 0)}")


def write_txt(samples, txt_path: Path):
    lines = [s["rel_img_path"] for s in samples]
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATASET_ROOT: {DATASET_ROOT}")

    print("\n开始检查 train split...")
    train_samples = collect_split(
        split_name="train",
        images_dir=IMAGES_TRAIN_DIR,
        labels_dir=LABELS_TRAIN_DIR,
    )

    print("\n开始检查 val split...")
    val_samples = collect_split(
        split_name="val",
        images_dir=IMAGES_VAL_DIR,
        labels_dir=LABELS_VAL_DIR,
    )

    summarize_split(train_samples, "train")
    summarize_split(val_samples, "val")

    write_txt(train_samples, TRAIN_TXT)
    write_txt(val_samples, VAL_TXT)

    print("\n已生成:")
    print(f"  {TRAIN_TXT}")
    print(f"  {VAL_TXT}")

    print("\n提示:")
    print("- 本脚本未移动任何图片或标签")
    print("- 本脚本不会重新划分 train / val")
    print("- data.yaml 应使用 train.txt / val.txt")


if __name__ == "__main__":
    main()
