from pathlib import Path
from ultralytics import YOLO


def main() -> None:
    # 数据集配置
    data_yaml = Path(
        r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\yolo_dataset1x_merged_classification\data.yaml"
    )

    # 训练输出目录
    project_dir = Path(
        r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs"
    )

    if not data_yaml.exists():
        raise FileNotFoundError(f"找不到 data.yaml: {data_yaml}")

    # 方案一：从官方预训练模型开始
    model = YOLO("yolov8n.pt")

    model.train(
        data=str(data_yaml),
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project=str(project_dir),
        name="yolov8n_1x_multiclass_v1",
        pretrained=True,
        patience=20,
        save=True,
        verbose=True,
    )


if __name__ == "__main__":
    main()