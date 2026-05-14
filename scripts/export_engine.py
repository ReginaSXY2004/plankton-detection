"""
Export YOLOv8 .pt model to TensorRT engine on Jetson.

This script does NOT overwrite best.pt.
It exports:

runs/yolov8n_1x_multiclass_v1/weights/best_jetson_fp16.engine

Run:
    PYTHONPATH=. python scripts/export_engine_jetson.py
"""

from pathlib import Path
import shutil
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]

WEIGHTS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "yolov8n_1x_multiclass_v1"
    / "weights"
)

PT_PATH = WEIGHTS_DIR / "best.pt"
TARGET_ENGINE_PATH = WEIGHTS_DIR / "best_jetson_fp16.engine"


def main():
    if not PT_PATH.exists():
        raise FileNotFoundError(f"找不到 pt 模型: {PT_PATH}")

    print("=" * 60)
    print("Export YOLO TensorRT Engine for Jetson")
    print(f"PT_PATH: {PT_PATH}")
    print(f"TARGET_ENGINE_PATH: {TARGET_ENGINE_PATH}")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    print(f"device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")
    else:
        raise RuntimeError("CUDA 不可用，不能导出 TensorRT engine")

    print("=" * 60)

    model = YOLO(str(PT_PATH))

    # Ultralytics 默认会导出到 best.engine
    exported_path = model.export(
        format="engine",
        imgsz=800,
        device=0,
        half=True,
        dynamic=False,
        simplify=True,
        workspace=4,
    )

    exported_path = Path(exported_path)

    if not exported_path.exists():
        raise RuntimeError(f"导出失败，未找到 engine: {exported_path}")

    # 不覆盖 best.pt，只把导出的 engine 重命名为新名字
    if exported_path.resolve() != TARGET_ENGINE_PATH.resolve():
        if TARGET_ENGINE_PATH.exists():
            print(f"已有旧 engine，删除: {TARGET_ENGINE_PATH}")
            TARGET_ENGINE_PATH.unlink()

        shutil.move(str(exported_path), str(TARGET_ENGINE_PATH))

    print("\n导出完成：")
    print(TARGET_ENGINE_PATH)


if __name__ == "__main__":
    main()