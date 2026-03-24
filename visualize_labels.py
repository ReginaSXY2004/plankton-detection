import os
import cv2

base_dir = r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\yolo_dataset1x"

images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")
output_dir = os.path.join(base_dir, "debug_vis")

os.makedirs(output_dir, exist_ok=True)

for img_name in os.listdir(images_dir):
    if not img_name.endswith((".jpg", ".png")):
        continue

    img_path = os.path.join(images_dir, img_name)
    label_path = os.path.join(labels_dir, img_name.rsplit(".", 1)[0] + ".txt")

    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 如果有标注文件
    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            for line in f:
                cls, x, y, bw, bh = map(float, line.strip().split())

                # YOLO格式 → 像素坐标
                x1 = int((x - bw/2) * w)
                y1 = int((y - bh/2) * h)
                x2 = int((x + bw/2) * w)
                y2 = int((y + bh/2) * h)

                # 画绿色框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # 类别（可选）
                cv2.putText(img, str(int(cls)), (x1, y1-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

    # 保存
    out_path = os.path.join(output_dir, img_name)
    cv2.imwrite(out_path, img)

print("全部生成完成！")