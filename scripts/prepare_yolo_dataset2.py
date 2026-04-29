from pathlib import Path
import random
from collections import Counter

# ========= 可改参数 =========
DATASET_ROOT = Path(
    r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\yolo_dataset1x_merged_classification"
)

IMAGES_DIR = DATASET_ROOT / "images" / "train"
LABELS_DIR = DATASET_ROOT / "labels" / "train"

TRAIN_TXT = DATASET_ROOT / "train.txt"
VAL_TXT = DATASET_ROOT / "val.txt"

VAL_RATIO = 0.15
RANDOM_SEED = 42

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
# ==========================


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMG_SUFFIXES


def get_label_path_from_image(img_path: Path) -> Path:
    return LABELS_DIR / f"{img_path.stem}.txt"


def parse_label_file(label_path: Path):
    """
    返回:
    - box_count: 该图总框数
    - class_counter: 该图各类框数 Counter
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


def collect_samples():
    """
    收集所有样本，返回 list[dict]
    每个元素包含:
    {
        'img_path': Path,
        'label_path': Path,
        'rel_img_path': str,
        'box_count': int,
        'class_counter': Counter,
        'is_empty': bool
    }
    """
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(f"图片目录不存在: {IMAGES_DIR}")
    if not LABELS_DIR.exists():
        raise FileNotFoundError(f"标签目录不存在: {LABELS_DIR}")

    image_files = sorted([p for p in IMAGES_DIR.iterdir() if is_image_file(p)])
    if not image_files:
        raise RuntimeError(f"没有在 {IMAGES_DIR} 找到图片")

    samples = []
    missing_labels = []

    for img_path in image_files:
        label_path = get_label_path_from_image(img_path)
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
            f"发现 {len(missing_labels)} 张图片缺少标签文件，前10个如下:\n{preview}"
        )

    return samples


def summarize_split(samples, title="split"):
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
    print("开始收集样本...")
    samples = collect_samples()

    print(f"总样本数: {len(samples)}")
    summarize_split(samples, "全量")

    rng = random.Random(RANDOM_SEED)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    val_count = int(len(shuffled) * VAL_RATIO)
    train_count = len(shuffled) - val_count

    train_samples = shuffled[:train_count]
    val_samples = shuffled[train_count:]

    if not train_samples or not val_samples:
        raise RuntimeError("train 或 val 为空，请检查 VAL_RATIO")

    write_txt(train_samples, TRAIN_TXT)
    write_txt(val_samples, VAL_TXT)

    print("\n已生成:")
    print(f"  {TRAIN_TXT}")
    print(f"  {VAL_TXT}")

    summarize_split(train_samples, "train")
    summarize_split(val_samples, "val")

    print("\n提示:")
    print(f"- 随机种子: {RANDOM_SEED}")
    print(f"- 验证集比例: {VAL_RATIO}")
    print("- 本脚本未移动任何图片或标签")


if __name__ == "__main__":
    main()