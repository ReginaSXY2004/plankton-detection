import random
import shutil
from pathlib import Path

# ========= 你可以改的参数 =========
DATASET_ROOT = Path(r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\yolo_dataset1x")
IMAGES_SRC = DATASET_ROOT / "images"
LABELS_SRC = DATASET_ROOT / "labels"

TRAIN_RATIO = 0.8
RANDOM_SEED = 42

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
# =================================


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_SUFFIXES


def main():
    random.seed(RANDOM_SEED)

    if not IMAGES_SRC.exists():
        raise FileNotFoundError(f"images 文件夹不存在: {IMAGES_SRC}")
    if not LABELS_SRC.exists():
        raise FileNotFoundError(f"labels 文件夹不存在: {LABELS_SRC}")

    # 找所有图片（只取当前层，不递归）
    image_files = [p for p in IMAGES_SRC.iterdir() if p.is_file() and is_image_file(p)]
    image_files.sort()

    if not image_files:
        raise RuntimeError(f"没有在 {IMAGES_SRC} 找到图片文件")

    print(f"共找到图片: {len(image_files)} 张")

    # 过滤掉没有标签的图
    paired = []
    missing_labels = []

    for img_path in image_files:
        label_path = LABELS_SRC / f"{img_path.stem}.txt"
        if label_path.exists():
            paired.append((img_path, label_path))
        else:
            missing_labels.append(img_path.name)

    print(f"有对应标签的图片: {len(paired)} 张")
    print(f"缺少标签的图片: {len(missing_labels)} 张")

    if missing_labels:
        print("\n以下图片没有找到对应 txt 标签，已跳过：")
        for name in missing_labels[:20]:
            print("  ", name)
        if len(missing_labels) > 20:
            print(f"  ... 还有 {len(missing_labels) - 20} 张")

    if len(paired) < 10:
        raise RuntimeError("有效带标签数据太少，暂时不建议开始训练。")

    random.shuffle(paired)

    train_count = int(len(paired) * TRAIN_RATIO)
    train_pairs = paired[:train_count]
    val_pairs = paired[train_count:]

    print(f"\n训练集: {len(train_pairs)} 张")
    print(f"验证集: {len(val_pairs)} 张")

    # 创建目标目录
    train_img_dir = DATASET_ROOT / "images" / "train"
    val_img_dir = DATASET_ROOT / "images" / "val"
    train_lbl_dir = DATASET_ROOT / "labels" / "train"
    val_lbl_dir = DATASET_ROOT / "labels" / "val"

    for d in [train_img_dir, val_img_dir, train_lbl_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # 检查是否已经划分过
    already_split = any(train_img_dir.iterdir()) or any(val_img_dir.iterdir())
    if already_split:
        print("\n警告：检测到 train/val 目录里已经有文件。")
        print("为了避免重复拷贝，脚本将直接退出。")
        print("如果你想重新划分，请先手动清空这些目录后再运行。")
        return

    # 复制文件
    def copy_pairs(pairs, dst_img_dir, dst_lbl_dir):
        for img_path, lbl_path in pairs:
            shutil.copy2(img_path, dst_img_dir / img_path.name)
            shutil.copy2(lbl_path, dst_lbl_dir / lbl_path.name)

    copy_pairs(train_pairs, train_img_dir, train_lbl_dir)
    copy_pairs(val_pairs, val_img_dir, val_lbl_dir)

    print("\n数据集划分完成。")

    # 生成 data.yaml
    yaml_path = DATASET_ROOT / "data.yaml"
    yaml_text = f"""path: {DATASET_ROOT.as_posix()}
train: images/train
val: images/val

names:
  0: microbe
"""

    yaml_path.write_text(yaml_text, encoding="utf-8")
    print(f"已生成 data.yaml: {yaml_path}")

    # 简单检查标签内容
    invalid_label_lines = []
    class_ids = set()

    for _, lbl_path in paired:
        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        for i, line in enumerate(lines, start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                invalid_label_lines.append((lbl_path.name, i, line))
                continue
            try:
                cls_id = int(float(parts[0]))
                nums = list(map(float, parts[1:]))
                class_ids.add(cls_id)
                # YOLO格式归一化坐标通常应在 0~1
                if not all(0.0 <= x <= 1.0 for x in nums):
                    invalid_label_lines.append((lbl_path.name, i, line))
            except Exception:
                invalid_label_lines.append((lbl_path.name, i, line))

    print(f"\n标签中出现的类别 id: {sorted(class_ids)}")

    if invalid_label_lines:
        print(f"\n发现疑似异常标签行: {len(invalid_label_lines)} 条")
        for item in invalid_label_lines[:20]:
            print("  文件:", item[0], "| 行号:", item[1], "| 内容:", item[2])
        if len(invalid_label_lines) > 20:
            print(f"  ... 还有 {len(invalid_label_lines) - 20} 条")
    else:
        print("标签格式检查通过。")

    print("\n下一步可以开始训练。")


if __name__ == "__main__":
    main()