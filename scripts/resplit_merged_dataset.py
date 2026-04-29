import shutil
from collections import defaultdict
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_ROOT = BASE_DIR / "data" / "yolo_dataset1x_merged"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMG_SUFFIXES


def sample_prefix(stem: str) -> str:
    parts = stem.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else stem


def frame_number(stem: str) -> int:
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return -1
    try:
        return int(parts[1])
    except ValueError:
        return -1


def collect_pairs():
    pairs = []
    missing = []

    for split in SPLITS:
        image_dir = DATASET_ROOT / "images" / split
        label_dir = DATASET_ROOT / "labels" / split

        for image_path in sorted(image_dir.glob("*")):
            if not image_path.is_file() or not is_image(image_path):
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            if label_path.exists():
                pairs.append((image_path, label_path))
            else:
                missing.append(image_path)

    return pairs, missing


def choose_split(index: int, total: int) -> str:
    if total < 3:
        return "train"

    train_end = max(1, int(round(total * TRAIN_RATIO)))
    val_count = max(1, int(round(total * VAL_RATIO)))
    val_end = min(total - 1, train_end + val_count)

    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def move_to_staging(pairs):
    staging_images = DATASET_ROOT / "_resplit_staging" / "images"
    staging_labels = DATASET_ROOT / "_resplit_staging" / "labels"

    if staging_images.parent.exists():
        raise RuntimeError(f"staging directory already exists: {staging_images.parent}")

    staging_images.mkdir(parents=True)
    staging_labels.mkdir(parents=True)

    staged = []
    for image_path, label_path in pairs:
        staged_image = staging_images / image_path.name
        staged_label = staging_labels / label_path.name
        shutil.move(str(image_path), str(staged_image))
        shutil.move(str(label_path), str(staged_label))
        staged.append((staged_image, staged_label))

    return staged


def write_split_txt(split: str, image_paths):
    txt_path = DATASET_ROOT / f"{split}.txt"
    lines = [
        path.relative_to(DATASET_ROOT).as_posix()
        for path in sorted(image_paths)
    ]
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def main():
    if not DATASET_ROOT.exists():
        raise FileNotFoundError(f"dataset not found: {DATASET_ROOT}")

    pairs, missing = collect_pairs()
    if missing:
        print(f"[warning] images missing labels: {len(missing)}")
        for path in missing[:20]:
            print(f"  {path}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    if not pairs:
        raise RuntimeError("no image/label pairs found.")

    staged_pairs = move_to_staging(pairs)

    groups = defaultdict(list)
    for image_path, label_path in staged_pairs:
        groups[sample_prefix(image_path.stem)].append((image_path, label_path))

    for group_pairs in groups.values():
        group_pairs.sort(key=lambda pair: frame_number(pair[0].stem))

    copied_by_split = {"train": [], "val": [], "test": []}
    group_summary = []

    for prefix in sorted(groups):
        group_pairs = groups[prefix]
        split_counts = {"train": 0, "val": 0, "test": 0}

        for index, (image_path, label_path) in enumerate(group_pairs):
            split = choose_split(index, len(group_pairs))
            dst_image_dir = DATASET_ROOT / "images" / split
            dst_label_dir = DATASET_ROOT / "labels" / split
            dst_image_dir.mkdir(parents=True, exist_ok=True)
            dst_label_dir.mkdir(parents=True, exist_ok=True)

            dst_image = dst_image_dir / image_path.name
            dst_label = dst_label_dir / label_path.name
            shutil.move(str(image_path), str(dst_image))
            shutil.move(str(label_path), str(dst_label))

            copied_by_split[split].append(dst_image)
            split_counts[split] += 1

        group_summary.append((prefix, split_counts))

    for split in SPLITS:
        write_split_txt(split, copied_by_split[split])

    staging_root = DATASET_ROOT / "_resplit_staging"
    for child in [staging_root / "images", staging_root / "labels", staging_root]:
        if child.exists():
            child.rmdir()

    print("resplit complete")
    print(f"dataset: {DATASET_ROOT}")
    for split in SPLITS:
        print(f"{split}: {len(copied_by_split[split])}")

    print("\nby sample:")
    for prefix, counts in group_summary:
        print(
            f"{prefix}: "
            f"train={counts['train']}, val={counts['val']}, test={counts['test']}"
        )


if __name__ == "__main__":
    main()
