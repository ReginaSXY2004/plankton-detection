"""
YOLOv8 多类别训练脚本

适用数据结构：

data/yolo_dataset1x_merged_classification/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── train.txt
├── val.txt
└── data.yaml

运行方式：
    python scripts/train_yolo2.py

如果从特殊环境运行，也可以：
    PYTHONPATH=. python scripts/train_yolo2.py
"""

from pathlib import Path
from ultralytics import YOLO


# scripts/train_yolo2.py -> 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "yolo_dataset1x_merged_classification"
DATA_YAML = DATASET_ROOT / "data.yaml"

PROJECT_DIR = PROJECT_ROOT / "runs"

RUN_NAME = "yolov8n_1x_multiclass_aug_v1" # 若开启数据增强，之前不开数据增强保存的是yolov8n_1x_multiclass_v1


def main() -> None:
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"找不到 data.yaml: {DATA_YAML}")

    if not (DATASET_ROOT / "images" / "train").exists():
        raise FileNotFoundError(f"找不到训练图片目录: {DATASET_ROOT / 'images' / 'train'}")

    if not (DATASET_ROOT / "images" / "val").exists():
        raise FileNotFoundError(f"找不到验证图片目录: {DATASET_ROOT / 'images' / 'val'}")

    if not (DATASET_ROOT / "labels" / "train").exists():
        raise FileNotFoundError(f"找不到训练标签目录: {DATASET_ROOT / 'labels' / 'train'}")

    if not (DATASET_ROOT / "labels" / "val").exists():
        raise FileNotFoundError(f"找不到验证标签目录: {DATASET_ROOT / 'labels' / 'val'}")

    if not (DATASET_ROOT / "train.txt").exists():
        raise FileNotFoundError(
            f"找不到 train.txt: {DATASET_ROOT / 'train.txt'}\n"
            "请先运行 scripts/prepare_yolo_dataset2.py"
        )

    if not (DATASET_ROOT / "val.txt").exists():
        raise FileNotFoundError(
            f"找不到 val.txt: {DATASET_ROOT / 'val.txt'}\n"
            "请先运行 scripts/prepare_yolo_dataset2.py"
        )

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    model = YOLO("yolov8n.pt")

    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        pretrained=True,
        patience=20,
        save=True,
        verbose=True,

        # 轻度数据增强参考：
        # degrees=5,
        # translate=0.05,
        # scale=0.25,
        # fliplr=0.5,
        # flipud=0.3,
        #
        # hsv_h=0.01,
        # hsv_s=0.25,
        # hsv_v=0.25,
        #
        # mosaic=0.15,
        # mixup=0.0,
    )


if __name__ == "__main__":
    main()
