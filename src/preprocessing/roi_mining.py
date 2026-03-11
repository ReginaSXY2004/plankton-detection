import cv2
from pathlib import Path
from preprocessing.background_subtraction import (
    create_background_subtractor,
    subtract_background
)

def extract_roi():
    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    video_path = BASE_DIR / "data" / "video" / "sample.avi"
    output_dir = BASE_DIR / "data" / "picture"
    output_dir.mkdir(parents=True, exist_ok=True)

    back_sub = create_background_subtractor()

    print("video path:", video_path)
    print("video exists:", video_path.exists())

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Error: video cannot be opened.")
        return

    frame_id = 0
    roi_id = 0

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1

        # 1. 背景减除
        fg_mask = subtract_background(back_sub, frame)

        # 2. 去噪
        blur = cv2.GaussianBlur(fg_mask, (5, 5), 0)

        # 3. 阈值分割
        _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

        if frame_id <= 5:
            debug_dir = BASE_DIR / "data" / "debug"
            debug_dir.mkdir(parents=True, exist_ok=True)

            cv2.imwrite(str(debug_dir / f"frame_{frame_id}.png"), frame)
            cv2.imwrite(str(debug_dir / f"fg_mask_{frame_id}.png"), fg_mask)
            cv2.imwrite(str(debug_dir / f"thresh_{frame_id}.png"), thresh)

        # 4. 形态学开运算，去掉小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        # 5. 找轮廓
        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 400 or area > 12000:
                continue

            x, y, w, h = cv2.boundingRect(cnt)

            # 避免裁太紧
            pad = 20
            x1 = max(x - pad, 0)
            y1 = max(y - pad, 0)
            x2 = min(x + w + pad, frame.shape[1])
            y2 = min(y + h + pad, frame.shape[0])

            roi = frame[y1:y2, x1:x2]

            save_path = output_dir / f"roi_{roi_id}.png"
            cv2.imwrite(str(save_path), roi)
            roi_id += 1

        print(f"frame {frame_id} processed")

    cap.release()
    print("Finished!")
    print("Total ROI saved:", roi_id)