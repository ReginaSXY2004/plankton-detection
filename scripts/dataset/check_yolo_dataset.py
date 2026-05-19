"""
check_yolo_dataset.py

功能：
    对已经整理好的 Ultralytics YOLO detection 数据集做训练前检查。

本脚本只读取和检查数据，不移动、不复制、不修改任何文件。

默认数据集结构：
    data/<dataset_name>/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    ├── data.yaml
    └── dataset_report.txt   # 可有可无，不影响检查

使用方式：
    放在：
        scripts/dataset/check_yolo_dataset.py

    Jetson运行：
        PYTHONPATH=. python3 scripts/dataset/check_yolo_dataset.py
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


# =========================
# CONFIG：主要改这里
# =========================

# 自动定位项目根目录：
# scripts/dataset/check_yolo_dataset.py -> project_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 要检查的干净 YOLO 数据集目录
DATASET_ROOT = PROJECT_ROOT / "data" / "yolo_dataset1x_classification_2"

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


# =========================
# 工具函数
# =========================

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_SUFFIXES


def load_class_names(yaml_path: Path) -> Dict[int, str]:
    """读取 data.yaml 中的类别映射，并检查 nc / names 是否一致。"""
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到 data.yaml: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise ValueError("data.yaml 内容不是合法 YAML 字典")

    if "names" not in data:
        raise ValueError("data.yaml 缺少 names 字段")

    names = data["names"]

    if isinstance(names, list):
        class_names = {i: str(name) for i, name in enumerate(names)}
    elif isinstance(names, dict):
        class_names = {int(k): str(v) for k, v in names.items()}
    else:
        raise ValueError("data.yaml 的 names 必须是 list 或 dict")

    if not class_names:
        raise ValueError("data.yaml 中 names 为空")

    if "nc" in data:
        nc = int(data["nc"])
        if nc != len(class_names):
            raise ValueError(
                f"data.yaml 中 nc={nc}，但 names 数量={len(class_names)}，两者不一致"
            )

    return class_names


def collect_images_and_labels(images_dir: Path, labels_dir: Path) -> Tuple[List[Path], List[Path]]:
    """收集一个 split 下的图片和 label txt。"""
    if not images_dir.exists():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")

    if not labels_dir.exists():
        raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

    images = sorted([p for p in images_dir.iterdir() if is_image(p)])
    labels = sorted([p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"])

    return images, labels


def check_pairing(split_name: str, images: List[Path], labels: List[Path]) -> Tuple[List[Path], List[Path]]:
    """
    检查 image-label 是否一一对应。

    missing_labels:
        有图片，但没有同名 txt。

    orphan_labels:
        有 txt，但没有同名图片。
    """
    image_stems = {p.stem for p in images}
    label_stems = {p.stem for p in labels}

    missing_label_stems = sorted(image_stems - label_stems)
    orphan_label_stems = sorted(label_stems - image_stems)

    image_by_stem = {p.stem: p for p in images}
    label_by_stem = {p.stem: p for p in labels}

    missing_labels = [image_by_stem[s] for s in missing_label_stems]
    orphan_labels = [label_by_stem[s] for s in orphan_label_stems]

    status = "PASS" if not missing_labels and not orphan_labels else "ERROR"
    print(f"\n[{status}] {split_name} image-label 对应检查")
    print(f"       images: {len(images)}")
    print(f"       labels: {len(labels)}")
    print(f"       missing labels: {len(missing_labels)}")
    print(f"       orphan labels: {len(orphan_labels)}")

    if missing_labels:
        print("       缺失 label 示例:")
        for p in missing_labels[:10]:
            print(f"       - {p}")

    if orphan_labels:
        print("       孤立 label 示例:")
        for p in orphan_labels[:10]:
            print(f"       - {p}")

    return missing_labels, orphan_labels


def parse_label_file(
    label_path: Path,
    valid_class_ids: set[int],
) -> Tuple[int, Counter, bool, List[str]]:
    """
    检查单个 YOLO detection label 文件。

    每行格式必须是：
        class_id x_center y_center width height

    后四个 bbox 数值必须是归一化坐标，即在 [0, 1] 范围内。
    """
    lines = label_path.read_text(encoding="utf-8").splitlines()

    box_count = 0
    class_counter = Counter()
    bad_lines: List[str] = []
    has_non_empty_line = False

    for line_no, line in enumerate(lines, start=1):
        raw = line.strip()

        if not raw:
            continue

        has_non_empty_line = True
        parts = raw.split()

        if len(parts) != 5:
            bad_lines.append(f"{label_path} | line {line_no} | 不是5列 | {raw}")
            continue

        try:
            cls_id = int(float(parts[0]))
        except Exception:
            bad_lines.append(f"{label_path} | line {line_no} | class id 不是整数 | {raw}")
            continue

        if cls_id not in valid_class_ids:
            bad_lines.append(
                f"{label_path} | line {line_no} | class id {cls_id} 不在 data.yaml names 中 | {raw}"
            )
            continue

        try:
            bbox = [float(x) for x in parts[1:]]
        except Exception:
            bad_lines.append(f"{label_path} | line {line_no} | bbox 不是数字 | {raw}")
            continue

        if not all(0.0 <= x <= 1.0 for x in bbox):
            bad_lines.append(f"{label_path} | line {line_no} | bbox 未归一化到 [0,1] | {raw}")
            continue

        box_count += 1
        class_counter[cls_id] += 1

    is_empty = not has_non_empty_line

    return box_count, class_counter, is_empty, bad_lines


def check_label_files(
    split_name: str,
    labels: List[Path],
    valid_class_ids: set[int],
) -> Tuple[Counter, List[Path], List[str], int]:
    """检查一个 split 下所有 label 文件内容，并统计类别分布。"""
    split_counter = Counter()
    empty_labels: List[Path] = []
    all_bad_lines: List[str] = []
    total_boxes = 0

    for label_path in labels:
        box_count, class_counter, is_empty, bad_lines = parse_label_file(
            label_path=label_path,
            valid_class_ids=valid_class_ids,
        )

        total_boxes += box_count
        split_counter.update(class_counter)

        if is_empty:
            empty_labels.append(label_path)

        all_bad_lines.extend(bad_lines)

    if all_bad_lines:
        status = "ERROR"
    elif empty_labels:
        status = "WARNING"
    else:
        status = "PASS"

    print(f"\n[{status}] {split_name} label 内容检查")
    print(f"       label files: {len(labels)}")
    print(f"       total boxes: {total_boxes}")
    print(f"       empty labels: {len(empty_labels)}")
    print(f"       bad lines: {len(all_bad_lines)}")

    if empty_labels:
        print("       空 label 示例:")
        for p in empty_labels[:10]:
            print(f"       - {p}")

    if all_bad_lines:
        print("       异常行示例:")
        for line in all_bad_lines[:10]:
            print(f"       - {line}")

    return split_counter, empty_labels, all_bad_lines, total_boxes


def check_train_val_overlap(train_images: List[Path], val_images: List[Path]) -> List[str]:
    """检查 train 和 val 是否有同名样本，避免数据泄漏。"""
    train_stems = {p.stem for p in train_images}
    val_stems = {p.stem for p in val_images}

    overlap = sorted(train_stems & val_stems)

    status = "PASS" if not overlap else "ERROR"
    print(f"\n[{status}] train/val 重复样本检查")
    print(f"       duplicate stems: {len(overlap)}")

    if overlap:
        print("       重复样本示例:")
        for stem in overlap[:20]:
            print(f"       - {stem}")

    return overlap


def print_class_distribution(class_names: Dict[int, str], total_counter: Counter) -> None:
    """打印每个类别的 bbox 数量。"""
    print("\nClass distribution / 类别统计")
    print("-" * 48)

    for cls_id in sorted(class_names):
        name = class_names[cls_id]
        count = total_counter.get(cls_id, 0)
        print(f"  class {cls_id:<2} {name:<20} : {count} boxes")


# =========================
# 主流程
# =========================

def main() -> None:
    yaml_path = DATASET_ROOT / "data.yaml"

    images_train_dir = DATASET_ROOT / "images" / "train"
    images_val_dir = DATASET_ROOT / "images" / "val"
    labels_train_dir = DATASET_ROOT / "labels" / "train"
    labels_val_dir = DATASET_ROOT / "labels" / "val"

    print("========== YOLO Dataset Check ==========")
    print(f"\nDataset root:\n  {DATASET_ROOT}")

    class_names = load_class_names(yaml_path)
    valid_class_ids = set(class_names.keys())

    print("\n[PASS] data.yaml loaded")
    print(f"       nc = {len(class_names)}")
    print("       classes:")
    for cls_id in sorted(class_names):
        print(f"       - {cls_id}: {class_names[cls_id]}")

    train_images, train_labels = collect_images_and_labels(images_train_dir, labels_train_dir)
    val_images, val_labels = collect_images_and_labels(images_val_dir, labels_val_dir)

    train_missing, train_orphan = check_pairing("train", train_images, train_labels)
    val_missing, val_orphan = check_pairing("val", val_images, val_labels)

    overlap = check_train_val_overlap(train_images, val_images)

    train_counter, train_empty, train_bad_lines, train_boxes = check_label_files(
        split_name="train",
        labels=train_labels,
        valid_class_ids=valid_class_ids,
    )

    val_counter, val_empty, val_bad_lines, val_boxes = check_label_files(
        split_name="val",
        labels=val_labels,
        valid_class_ids=valid_class_ids,
    )

    total_counter = Counter()
    total_counter.update(train_counter)
    total_counter.update(val_counter)

    print_class_distribution(class_names, total_counter)

    error_count = (
        len(train_missing)
        + len(train_orphan)
        + len(val_missing)
        + len(val_orphan)
        + len(overlap)
        + len(train_bad_lines)
        + len(val_bad_lines)
    )

    print("\nOverall summary / 总体统计")
    print("-" * 48)
    print(f"  train images: {len(train_images)}")
    print(f"  val images: {len(val_images)}")
    print(f"  train boxes: {train_boxes}")
    print(f"  val boxes: {val_boxes}")
    print(f"  empty labels total: {len(train_empty) + len(val_empty)}")
    print(f"  errors total: {error_count}")

    if error_count > 0:
        print("\n========== CHECK FAILED ==========")
        print("数据集存在严重问题，请修复后再训练。")
        raise RuntimeError("YOLO dataset check failed.")

    print("\n========== CHECK PASSED ==========")
    print("数据集结构和 label 格式检查通过，可以开始 YOLO 训练。")


if __name__ == "__main__":
    main()
