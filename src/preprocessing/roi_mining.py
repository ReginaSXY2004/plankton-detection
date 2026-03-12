import cv2
from pathlib import Path


def extract_roi():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    video_path = BASE_DIR / "data" / "video" / "sample.avi"
    output_dir = BASE_DIR / "data" / "picture_gray"
    debug_dir = BASE_DIR / "data" / "debug_gray"

    output_dir.mkdir(parents=True, exist_ok=True)
    debug_dir.mkdir(parents=True, exist_ok=True)

    print("video path:", video_path)
    print("video exists:", video_path.exists())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: video cannot be opened.")
        return

    frame_id = 0
    roi_id = 0

    # 先只跑前300帧，避免一次生成太多ROI
    max_frames = 300

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        if frame_id > max_frames:
            break

        # 1. 灰度化
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 2. 轻度去噪
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 3. 阈值分割
        # 黑背景 + 亮目标，先从 35 试起
        _, thresh = cv2.threshold(blur, 35, 255, cv2.THRESH_BINARY)

        # 4. 形态学闭运算，把断开的亮边连起来
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 可选：再做一次开运算，去掉很小的噪点
        small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, small_kernel)

        # 保存前5帧调试图
        if frame_id <= 5:
            cv2.imwrite(str(debug_dir / f"frame_{frame_id}.png"), frame)
            cv2.imwrite(str(debug_dir / f"gray_{frame_id}.png"), gray)
            cv2.imwrite(str(debug_dir / f"thresh_{frame_id}.png"), thresh)

        # 5. 找轮廓
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        frame_h, frame_w = frame.shape[:2]

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # 6. 面积过滤
            if area < 500 or area > 15000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # 7. bbox 尺寸过滤
            if w < 35 or h < 35:
                continue
            if w > 220 or h > 220:
                continue

            # 8. 宽高比过滤（先放宽一点）
            aspect_ratio = w / h if h != 0 else 0
            if aspect_ratio < 0.2 or aspect_ratio > 5:
                continue

            # 9. 边缘过滤：去掉贴边的半只目标
            if x <= 3 or y <= 3 or x + w >= frame_w - 3 or y + h >= frame_h - 3:
                continue

            # 10. 外扩一点，避免裁太紧
            pad = 18
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, frame_w)
            y2 = min(y + h + pad, frame_h)

            roi = frame[y1:y2, x1:x2]

            save_path = output_dir / f"roi_{roi_id}.png"
            cv2.imwrite(str(save_path), roi)
            roi_id += 1

        print(f"frame {frame_id} processed")

    cap.release()
    print("Finished!")
    print("Total ROI saved:", roi_id)