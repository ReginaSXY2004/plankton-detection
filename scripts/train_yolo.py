from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")  # 先从最小模型开始

    model.train(
        data=r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\yolo_dataset1x\data.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0,
        project=r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\runs",
        name="yolov8n_microbe_1x",
        pretrained=True,
        patience=20,
        save=True
    )


if __name__ == "__main__":
    main()