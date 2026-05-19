"""
build_clean_yolo_dataset.py

功能：
    将任意乱结构的 CVAT / YOLO 导出数据集，整理成标准 Ultralytics YOLO
    detection 训练数据集。

最终输出结构：
    data/<dataset_name>/
    ├── images/
    │   ├── train/
    │   └── val/
    ├── labels/
    │   ├── train/
    │   └── val/
    ├── data.yaml
    └── dataset_report.txt

重要原则：
    1. 不修改原始数据，只复制到新的干净数据集。
    2. 不猜类别、不 remap class id、不修改 label 类别。
    3. 类别映射由人工写好的 data.yaml 决定。
    4. 图片和 label 按文件名 stem 匹配：
       sample15_000005.png <-> sample15_000005.txt
    5. 允许空 txt，记录为空图。
    6. 缺失 label 默认视为错误，避免把漏标图片误当成负样本。

使用方式：
    放在：
        scripts/dataset/build_clean_yolo_dataset.py

    Jetson运行：
        cd ~/plankton-detection
        source venv_jetson/bin/activate
        
        PYTHONPATH=. python3 scripts/dataset/build_clean_yolo_dataset.py
"""

from __future__ import annotations

import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml


# =========================
# CONFIG
# =========================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 原始乱结构数据目录，比如 CVAT 导出的 zip 解压后的文件夹。
# 这个目录可以很乱，脚本会递归往最深处找图片和 txt。
SOURCE_DIR = PROJECT_ROOT / "data" / "1111111" # 改为人工标注后重新上传上来的数据集的路径

SOURCE_DATA_YAML = SOURCE_DIR / "data.yaml"

# 输出的干净 YOLO 训练数据集。
# data.yaml 最终也会被写入这个目录里面。
OUTPUT_DIR = PROJECT_ROOT / "data" / "yolo_dataset1x_classification_2"

VAL_RATIO = 0.2
RANDOM_SEED = 42

# 为了安全，默认不覆盖已有输出目录。
# 如果确认要重新生成，把它改成 True。
OVERWRITE_OUTPUT = False

# 如果 False：发现图片没有对应 label 就停止。
# 如果 True：跳过这些图片，并写入 report。
SKIP_MISSING_LABELS = False

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 这些 txt 不是 YOLO label，递归扫描时必须排除。
NON_LABEL_TXT_NAMES = {
    "train.txt",
    "val.txt",
    "test.txt",
    "dataset_report.txt",
    "classes.txt",
    "obj.names",
    "obj.data",
    "README.txt",
    "readme.txt",
}


# =========================
# 数据结构
# =========================

@dataclass(frozen=True)
class Pair:
    image_path: Path
    label_path: Path
    stem: str
    box_count: int
    class_counter: Counter
    is_empty: bool


@dataclass
class BuildStats:
    total_images_found: int = 0
    total_label_files_found: int = 0
    matched_pairs: int = 0
    train_count: int = 0
    val_count: int = 0
    empty_label_count: int = 0
    missing_label_count: int = 0
    orphan_label_count: int = 0
    duplicate_image_stems: Dict[str, List[Path]] = None
    duplicate_label_stems: Dict[str, List[Path]] = None
    missing_labels: List[Path] = None
    orphan_labels: List[Path] = None
    empty_labels: List[Path] = None
    bad_label_lines: List[str] = None
    class_counter: Counter = None

    def __post_init__(self):
        self.duplicate_image_stems = self.duplicate_image_stems or {}
        self.duplicate_label_stems = self.duplicate_label_stems or {}
        self.missing_labels = self.missing_labels or []
        self.orphan_labels = self.orphan_labels or []
        self.empty_labels = self.empty_labels or []
        self.bad_label_lines = self.bad_label_lines or []
        self.class_counter = self.class_counter or Counter()


# =========================
# 基础工具函数
# =========================

def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_SUFFIXES


def is_label_txt(path: Path) -> bool:
    """判断一个 txt 是否应该被当作 YOLO label 文件。"""
    if not path.is_file():
        return False
    if path.suffix.lower() != ".txt":
        return False
    return path.name not in NON_LABEL_TXT_NAMES


def collect_by_stem(paths: Iterable[Path]) -> Tuple[Dict[str, Path], Dict[str, List[Path]]]:
    """
    建立 stem -> path 映射，并检查重名 stem。

    如果乱结构里出现两个 sample15_000005.png，脚本无法判断哪个 label
    应该配哪个 image，所以必须报警停止。
    """
    grouped: Dict[str, List[Path]] = defaultdict(list)

    for path in paths:
        grouped[path.stem].append(path)

    unique: Dict[str, Path] = {}
    duplicates: Dict[str, List[Path]] = {}

    for stem, items in grouped.items():
        if len(items) == 1:
            unique[stem] = items[0]
        else:
            duplicates[stem] = sorted(items)

    return unique, duplicates


def load_class_names(yaml_path: Path) -> Dict[int, str]:
    """
    从人工编写的 data.yaml 读取类别映射。

    支持两种 names 写法：
        names:
          0: daxingzao
          1: jianshuizao

    或：
        names: ["daxingzao", "jianshuizao"]
    """
    if not yaml_path.exists():
        raise FileNotFoundError(f"找不到人工 data.yaml: {yaml_path}")

    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict) or "names" not in data:
        raise ValueError("data.yaml 必须包含 names 字段")

    names = data["names"]

    if isinstance(names, list):
        class_names = {i: str(name) for i, name in enumerate(names)}
    elif isinstance(names, dict):
        class_names = {int(k): str(v) for k, v in names.items()}
    else:
        raise ValueError("data.yaml 的 names 必须是 list 或 dict")

    if not class_names:
        raise ValueError("data.yaml 的 names 为空")

    if "nc" in data and int(data["nc"]) != len(class_names):
        raise ValueError(
            f"data.yaml 中 nc={data['nc']}，但 names 数量={len(class_names)}，两者不一致"
        )

    return class_names


def parse_label_file(
    label_path: Path,
    valid_class_ids: set[int],
    stats: BuildStats,
) -> Tuple[int, Counter, bool]:
    """
    解析并检查单个 YOLO detection label 文件。

    每行必须是：
        class_id x_center y_center width height

    后四个 bbox 数值必须是归一化坐标，也就是在 [0, 1] 之间。
    空 txt 合法，会被记录为空图。
    """
    text = label_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    class_counter = Counter()
    box_count = 0
    has_non_empty_line = False

    for line_no, line in enumerate(lines, start=1):
        raw = line.strip()
        if not raw:
            continue

        has_non_empty_line = True
        parts = raw.split()

        if len(parts) != 5:
            stats.bad_label_lines.append(
                f"{label_path} | line {line_no} | 不是5列 | {raw}"
            )
            continue

        try:
            cls_id = int(float(parts[0]))
        except Exception:
            stats.bad_label_lines.append(
                f"{label_path} | line {line_no} | class id 不是整数 | {raw}"
            )
            continue

        if cls_id not in valid_class_ids:
            stats.bad_label_lines.append(
                f"{label_path} | line {line_no} | class id {cls_id} 不在 data.yaml names 中 | {raw}"
            )
            continue

        try:
            bbox = [float(x) for x in parts[1:]]
        except Exception:
            stats.bad_label_lines.append(
                f"{label_path} | line {line_no} | bbox 不是数字 | {raw}"
            )
            continue

        if not all(0.0 <= x <= 1.0 for x in bbox):
            stats.bad_label_lines.append(
                f"{label_path} | line {line_no} | bbox 未归一化到 [0,1] | {raw}"
            )
            continue

        box_count += 1
        class_counter[cls_id] += 1

    is_empty = not has_non_empty_line

    if is_empty:
        stats.empty_labels.append(label_path)

    return box_count, class_counter, is_empty


def ensure_clean_output_dir(output_dir: Path) -> None:
    """创建干净输出目录。默认不覆盖，防止误删已有训练数据。"""
    if output_dir.exists():
        if not OVERWRITE_OUTPUT:
            raise FileExistsError(
                f"输出目录已存在: {output_dir}\n"
                f"为避免覆盖数据，脚本已停止。\n"
                f"如果确认要重新生成，请设置 OVERWRITE_OUTPUT=True"
            )
        shutil.rmtree(output_dir)

    (output_dir / "images" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "images" / "val").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (output_dir / "labels" / "val").mkdir(parents=True, exist_ok=True)


def copy_pairs(pairs: List[Pair], split: str, output_dir: Path) -> None:
    """复制 image-label 对到标准 YOLO 目录。"""
    image_out_dir = output_dir / "images" / split
    label_out_dir = output_dir / "labels" / split

    for pair in pairs:
        dst_img = image_out_dir / pair.image_path.name
        dst_label = label_out_dir / f"{pair.stem}.txt"

        shutil.copy2(pair.image_path, dst_img)
        shutil.copy2(pair.label_path, dst_label)


def write_clean_data_yaml(output_dir: Path, class_names: Dict[int, str]) -> None:
    """
    在输出数据集目录中写入最终训练用 data.yaml。

    这里会强制使用相对路径：
        train: images/train
        val: images/val

    这样 Windows 和 Jetson 都能直接使用。
    """
    lines = [
        "path: .",
        "train: images/train",
        "val: images/val",
        "",
        f"nc: {len(class_names)}",
        "names:",
    ]

    for cls_id in sorted(class_names):
        lines.append(f"  {cls_id}: {class_names[cls_id]}")

    (output_dir / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_path_list(title: str, paths: List[Path], max_items: int = 50) -> List[str]:
    out = [f"\n{title}: {len(paths)}"]
    for p in paths[:max_items]:
        out.append(f"  {p}")
    if len(paths) > max_items:
        out.append(f"  ... 还有 {len(paths) - max_items} 个未显示")
    return out


def write_report(
    output_dir: Path,
    source_dir: Path,
    source_yaml: Path,
    class_names: Dict[int, str],
    stats: BuildStats,
) -> None:
    """生成数据集检查报告。该文件不会影响 YOLO 训练。"""
    report_path = output_dir / "dataset_report.txt"

    lines = []
    lines.append("YOLO Dataset Build Report")
    lines.append("=" * 36)
    lines.append(f"生成时间: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"原始数据目录: {source_dir}")
    lines.append(f"输出目录: {output_dir}")
    lines.append(f"原始 data.yaml: {source_yaml}")
    lines.append("")
    lines.append("数据集概要")
    lines.append("-" * 36)
    lines.append(f"Total images found: {stats.total_images_found}")
    lines.append(f"Total label txt files found: {stats.total_label_files_found}")
    lines.append(f"Matched image-label pairs: {stats.matched_pairs}")
    lines.append(f"Missing labels: {stats.missing_label_count}")
    lines.append(f"Orphan labels: {stats.orphan_label_count}")
    lines.append(f"空 label 文件: {stats.empty_label_count}")
    lines.append(f"异常 label 行: {len(stats.bad_label_lines)}")
    lines.append(f"Train images: {stats.train_count}")
    lines.append(f"Val images: {stats.val_count}")
    lines.append("")

    lines.append("类别统计")
    lines.append("-" * 36)
    for cls_id in sorted(class_names):
        lines.append(
            f"class {cls_id}: {class_names[cls_id]:<20} -> "
            f"{stats.class_counter.get(cls_id, 0)} boxes"
        )

    if stats.duplicate_image_stems:
        lines.append("\n重复图片 stem")
        lines.append("-" * 36)
        for stem, paths in list(stats.duplicate_image_stems.items())[:50]:
            lines.append(f"{stem}:")
            for p in paths:
                lines.append(f"  {p}")

    if stats.duplicate_label_stems:
        lines.append("\n重复 label stem")
        lines.append("-" * 36)
        for stem, paths in list(stats.duplicate_label_stems.items())[:50]:
            lines.append(f"{stem}:")
            for p in paths:
                lines.append(f"  {p}")

    lines.extend(format_path_list("缺失 label 的图片", stats.missing_labels))
    lines.extend(format_path_list("孤立 label 文件", stats.orphan_labels))
    lines.extend(format_path_list("空 label 文件", stats.empty_labels))

    if stats.bad_label_lines:
        lines.append("\n异常 label 行")
        lines.append("-" * 36)
        for item in stats.bad_label_lines[:100]:
            lines.append(item)
        if len(stats.bad_label_lines) > 100:
            lines.append(f"... 还有 {len(stats.bad_label_lines) - 100} 条未显示")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# =========================
# 主流程
# =========================

def main() -> None:
    if not SOURCE_DIR.exists():
        raise FileNotFoundError(f"SOURCE_DIR 不存在: {SOURCE_DIR}")

    if not 0.0 < VAL_RATIO < 1.0:
        raise ValueError("VAL_RATIO 必须在 0 到 1 之间")

    class_names = load_class_names(SOURCE_DATA_YAML)
    valid_class_ids = set(class_names.keys())

    stats = BuildStats()

    # 递归扫描：不相信目录层级，只相信文件本身和 stem。
    image_files = sorted([p for p in SOURCE_DIR.rglob("*") if is_image(p)])
    label_files = sorted([p for p in SOURCE_DIR.rglob("*") if is_label_txt(p)])

    stats.total_images_found = len(image_files)
    stats.total_label_files_found = len(label_files)

    image_map, image_duplicates = collect_by_stem(image_files)
    label_map, label_duplicates = collect_by_stem(label_files)

    stats.duplicate_image_stems = image_duplicates
    stats.duplicate_label_stems = label_duplicates

    if image_duplicates or label_duplicates:
        ensure_clean_output_dir(OUTPUT_DIR)
        write_report(OUTPUT_DIR, SOURCE_DIR, SOURCE_DATA_YAML, class_names, stats)
        raise RuntimeError(
            "发现重复 stem，无法安全匹配 image-label。\n"
            f"详情见: {OUTPUT_DIR / 'dataset_report.txt'}"
        )

    pairs: List[Pair] = []

    for stem, img_path in image_map.items():
        label_path = label_map.get(stem)

        if label_path is None:
            stats.missing_labels.append(img_path)
            continue

        box_count, class_counter, is_empty = parse_label_file(
            label_path=label_path,
            valid_class_ids=valid_class_ids,
            stats=stats,
        )

        pairs.append(
            Pair(
                image_path=img_path,
                label_path=label_path,
                stem=stem,
                box_count=box_count,
                class_counter=class_counter,
                is_empty=is_empty,
            )
        )

        stats.class_counter.update(class_counter)

    for stem, label_path in label_map.items():
        if stem not in image_map:
            stats.orphan_labels.append(label_path)

    stats.matched_pairs = len(pairs)
    stats.missing_label_count = len(stats.missing_labels)
    stats.orphan_label_count = len(stats.orphan_labels)
    stats.empty_label_count = len(stats.empty_labels)

    # 缺失 label 默认停止，避免把漏标图片当成空图。
    if stats.missing_labels and not SKIP_MISSING_LABELS:
        ensure_clean_output_dir(OUTPUT_DIR)
        write_report(OUTPUT_DIR, SOURCE_DIR, SOURCE_DATA_YAML, class_names, stats)
        raise RuntimeError(
            "发现图片缺少对应 label txt，脚本已停止。\n"
            f"详情见: {OUTPUT_DIR / 'dataset_report.txt'}"
        )

    # label 内容异常时停止，不输出可能污染训练的数据集。
    if stats.bad_label_lines:
        ensure_clean_output_dir(OUTPUT_DIR)
        write_report(OUTPUT_DIR, SOURCE_DIR, SOURCE_DATA_YAML, class_names, stats)
        raise RuntimeError(
            "发现异常 label 行，脚本已停止。\n"
            f"详情见: {OUTPUT_DIR / 'dataset_report.txt'}"
        )

    if not pairs:
        ensure_clean_output_dir(OUTPUT_DIR)
        write_report(OUTPUT_DIR, SOURCE_DIR, SOURCE_DATA_YAML, class_names, stats)
        raise RuntimeError(
            "没有找到任何成功匹配的 image-label pair。\n"
            f"详情见: {OUTPUT_DIR / 'dataset_report.txt'}"
        )

    random.seed(RANDOM_SEED)
    random.shuffle(pairs)

    val_count = max(1, int(round(len(pairs) * VAL_RATIO))) if len(pairs) > 1 else 0
    val_pairs = pairs[:val_count]
    train_pairs = pairs[val_count:]

    stats.train_count = len(train_pairs)
    stats.val_count = len(val_pairs)

    ensure_clean_output_dir(OUTPUT_DIR)

    copy_pairs(train_pairs, "train", OUTPUT_DIR)
    copy_pairs(val_pairs, "val", OUTPUT_DIR)
    write_clean_data_yaml(OUTPUT_DIR, class_names)
    write_report(OUTPUT_DIR, SOURCE_DIR, SOURCE_DATA_YAML, class_names, stats)

    print("\n========== YOLO 数据集整理完成 ==========")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"训练集图片数: {stats.train_count}")
    print(f"验证集图片数: {stats.val_count}")
    print(f"空 label 数量: {stats.empty_label_count}")
    print(f"报告文件: {OUTPUT_DIR / 'dataset_report.txt'}")
    print("\n下一步可训练：")
    print(f"  yolo detect train data={OUTPUT_DIR / 'data.yaml'} model=yolov8n.pt imgsz=800")


if __name__ == "__main__":
    main()
