from pathlib import Path
import cv2
from typing import Dict, Tuple, List

BASE_DIR = Path(r"C:\Users\Regina Sun\Documents\GitHub\plankton-detection\data\video1xpic")
IMAGES_DIR = BASE_DIR / "images" / "train" / "images"
LABELS_DIR = BASE_DIR / "labels" / "train" / "images"
OUTPUT_DIR = BASE_DIR / "vis"
SHOW_LABEL = True
LINE_THICKNESS = 2
FONT_SCALE = 0.6

CLASS_NAMES = {
    0: "Da Xing Zao",
    1: "Jian Shui Zao",
    2: "Xian Nv Chong",
    3: "Lun Chong",
    4: "Xiang Bi Zao",
    5: "Unkown",
}

CLASS_COLORS = {
    0: (112, 224, 131),  # #83E070
    1: (245, 61, 61),    # #3D3DF5
    2: (51, 204, 255),   # #FFCC33
    3: (183, 50, 250),   # #FA32B7
    4: (209, 240, 170),  # #AAF0D1
    5: (80, 80, 178),    # #B25050
}
# =================


def yolo_to_xyxy(xc, yc, w, h, img_w, img_h):
    x1 = int((xc - w / 2) * img_w)
    y1 = int((yc - h / 2) * img_h)
    x2 = int((xc + w / 2) * img_w)
    y2 = int((yc + h / 2) * img_h)

    x1 = max(0, min(x1, img_w - 1))
    y1 = max(0, min(y1, img_h - 1))
    x2 = max(0, min(x2, img_w - 1))
    y2 = max(0, min(y2, img_h - 1))
    return x1, y1, x2, y2


def read_yolo_txt(label_path: Path):
    results = []
    if not label_path.exists():
        return results

    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            parts = line.strip().split()
            if len(parts) != 5:
                print(f"[warning] 跳过格式错误: {label_path.name}:{line_num}")
                continue

            try:
                cls_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:])
                results.append((cls_id, xc, yc, w, h))
            except Exception:
                print(f"[warning] 跳过无法解析行: {label_path.name}:{line_num}")
    return results


def draw_boxes(img, boxes):
    h, w = img.shape[:2]

    for cls_id, xc, yc, bw, bh in boxes:
        x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = CLASS_NAMES.get(cls_id, str(cls_id))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, LINE_THICKNESS)

        if SHOW_LABEL:
            text = label
            (tw, th), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, 1
            )
            box_y1 = max(0, y1 - th - baseline - 4)
            box_y2 = box_y1 + th + baseline + 4
            box_x2 = min(w - 1, x1 + tw + 6)

            cv2.rectangle(img, (x1, box_y1), (box_x2, box_y2), color, -1)
            cv2.putText(
                img,
                text,
                (x1 + 3, box_y2 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                FONT_SCALE,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

    return img


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        [p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}]
    )

    print(f"images dir: {IMAGES_DIR}")
    print(f"labels dir: {LABELS_DIR}")
    print(f"找到图片 {len(image_paths)} 张")

    for idx, img_path in enumerate(image_paths, start=1):
        label_path = LABELS_DIR / f"{img_path.stem}.txt"

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"[warning] 读图失败: {img_path}")
            continue

        boxes = read_yolo_txt(label_path)
        vis = draw_boxes(img.copy(), boxes)

        out_path = OUTPUT_DIR / img_path.name
        cv2.imwrite(str(out_path), vis)

        if idx % 50 == 0 or idx == len(image_paths):
            print(f"已处理 {idx}/{len(image_paths)}")

    print(f"完成，结果在: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()