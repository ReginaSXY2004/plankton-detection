"""
train_yolo.py

功能：
    使用 Ultralytics YOLOv8 训练微生物多类别检测模型。

要求：
    1. 已经准备好标准 YOLO 数据集：
        data/<DATASET_NAME>/
        ├── images/
        │   ├── train/xxx.png
        │   └── val/xxx.png
        ├── labels/
        │   ├── train/xxx.txt
        │   └── val/xxx.txt
        └── data.yaml

    2. 从项目根目录运行：
        python scripts/train/train_yolo.py
        或者
        PYTHONPATH=. python scripts/train/train_yolo.py

"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ultralytics import YOLO


# =========================
# CONFIG：主要改这里
# =========================

# scripts/train/train_yolo.py -> project_root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 训练使用的数据集名称，对应 data/ 下的文件夹名
DATASET_NAME = "yolo_dataset1x_v2"

# 本次训练输出名称，对应 runs/<RUN_NAME>/
RUN_NAME = "yolov8n_1x_multiclass_v2"

# 预训练模型
MODEL_NAME = "yolov8n.pt"

# 训练参数
EPOCHS = 100
IMGSZ = 800
BATCH = 16
WORKERS = 4
DEVICE = 0
PATIENCE = 20


# =========================
# 路径
# =========================

DATASET_ROOT = PROJECT_ROOT / "data" / DATASET_NAME
DATA_YAML = DATASET_ROOT / "data.yaml"
PROJECT_DIR = PROJECT_ROOT / "runs"


def check_dataset() -> None:
    """训练前做最基本的目录检查。更完整检查请运行 check_yolo_dataset.py。"""
    required_paths = [
        DATA_YAML,
        DATASET_ROOT / "images" / "train",
        DATASET_ROOT / "images" / "val",
        DATASET_ROOT / "labels" / "train",
        DATASET_ROOT / "labels" / "val",
    ]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"找不到必要路径: {path}")


def save_training_config(run_dir: Path) -> None:
    """保存本次训练配置，方便之后复现实验。"""
    config_text = f"""YOLO Training Config
====================
time: {datetime.now().isoformat(timespec="seconds")}

dataset_name: {DATASET_NAME}
dataset_root: {DATASET_ROOT}
data_yaml: {DATA_YAML}

model_name: {MODEL_NAME}
run_name: {RUN_NAME}

epochs: {EPOCHS}
imgsz: {IMGSZ}
batch: {BATCH}
workers: {WORKERS}
device: {DEVICE}
patience: {PATIENCE}
"""

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "training_config.txt").write_text(config_text, encoding="utf-8")


def main() -> None:
    check_dataset()

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    run_dir = PROJECT_DIR / RUN_NAME
    save_training_config(run_dir)

    model = YOLO(MODEL_NAME)

    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        workers=WORKERS,
        device=DEVICE,
        project=str(PROJECT_DIR),
        name=RUN_NAME,
        pretrained=True,
        patience=PATIENCE,
        save=True,
        verbose=True,
    )

    print("\n========== 训练完成 ==========")
    print(f"训练结果目录: {run_dir}")
    print(f"最佳模型路径: {run_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
