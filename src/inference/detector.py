from ultralytics import YOLO

from src.tracking.microbe_tracker import Detection


class YoloDetector:
    def __init__(self, model_path, device=0):
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, frame, conf: float, imgsz: int):
        result = self.model.predict(
            source=frame,
            conf=conf,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )[0]

        detections = []

        if result.boxes is None or len(result.boxes) == 0:
            return detections

        xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        clses = result.boxes.cls.cpu().numpy().astype(int)

        for i, box in enumerate(xyxy):
            x1, y1, x2, y2 = map(float, box)

            detections.append(
                Detection(
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    conf=float(confs[i]),
                    cls_id=int(clses[i])
                )
            )

        return detections