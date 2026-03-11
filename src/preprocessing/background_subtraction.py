import cv2


def create_background_subtractor():
    """
    创建背景减除器
    """
    back_sub = cv2.createBackgroundSubtractorMOG2(
        history=2000,
        varThreshold=25,
        detectShadows=False
    )
    return back_sub


def subtract_background(back_sub, frame):
    """
    对一帧图像做背景减除
    """
    fg_mask = back_sub.apply(frame)

    return fg_mask