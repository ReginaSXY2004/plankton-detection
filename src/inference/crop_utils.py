"""
best crop 与图像质量工具函数

本文件负责：
1. 安全裁剪 bbox 区域
2. 计算 crop 清晰度
3. 判断 crop 是否有资格参与 best crop 竞争
4. 计算 best crop 分数
5. 将秒数转换为帧数
6. 根据类别投票确定轨迹最终类别

这些函数不直接负责保存图片，
只负责判断“哪一帧的 crop 更适合作为该 track 的代表图”。
"""

import cv2
import numpy as np


from src.config.infer_config import (
    BEST_MIN_W,
    BEST_MIN_H,
    BEST_MIN_CONF,
    BEST_MIN_SHARPNESS,
    MIN_CLASS_VOTES_TO_LOCK,
)


def compute_sharpness(roi_bgr: np.ndarray) -> float:
    """
    计算 crop 的清晰度。
    """

    if roi_bgr is None or roi_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)

    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def safe_crop(frame: np.ndarray, bbox):
    """
    根据 bbox 从原始帧中安全裁剪 crop。

    自动处理：
    1. bbox 超出图像边界
    2. bbox 坐标不是整数
    3. bbox 无效导致裁剪区域为空

    返回：
    crop: 裁剪出的图像区域；如果无效则为 None
    clipped_bbox: 裁剪后的合法 bbox
    """

    h, w = frame.shape[:2]

    x1, y1, x2, y2 = bbox

    x1 = max(0, min(int(round(x1)), w - 1))
    y1 = max(0, min(int(round(y1)), h - 1))
    x2 = max(0, min(int(round(x2)), w))
    y2 = max(0, min(int(round(y2)), h))

    if x2 <= x1 or y2 <= y1:
        return None, (x1, y1, x2, y2)

    crop = frame[y1:y2, x1:x2]

    return crop, (x1, y1, x2, y2)


def check_best_candidate(tr, crop: np.ndarray, sharpness: float):
    """
    判断当前 track 的 crop 是否有资格参与 best crop 竞争。

    过滤原因包括：
    1. crop 为空
    2. crop 尺寸过小
    3. 当前 track 置信度过低
    4. crop 清晰度过低

    返回：
    ok: 是否合格
    reason: 不合格原因，或 "ok"
    """

    if crop is None or crop.size == 0:
        return False, "empty_crop"

    h, w = crop.shape[:2]

    if w < BEST_MIN_W or h < BEST_MIN_H:
        return False, "too_small"

    if tr.conf < BEST_MIN_CONF:
        return False, "low_conf"

    if sharpness < BEST_MIN_SHARPNESS:
        return False, "low_sharpness"

    return True, "ok"


def best_score(sharpness: float, conf: float, area: float) -> float:
    """
    计算 best crop 综合得分。

    当前评分综合考虑：
    1. 清晰度
    2. detection 置信度
    3. bbox 面积

    面积项会被限制上限，避免大框天然占优势。
    """

    return (
        sharpness * 1.0
        + conf * 120.0
        + min(area, 2500.0) * 0.01
    )


def seconds_to_frames(
    seconds: float,
    fps: float,
    min_frames: int = 1
) -> int:
    """
    将秒数转换为帧数。

    用途：
    让 confirm / missed / reconnect 等 tracker 阈值
    不再强依赖具体视频 FPS。
    """

    return max(min_frames, int(round(seconds * fps)))


def majority_class_from_votes(class_votes):
    """
    根据轨迹历史类别投票，确定该 track 的最终类别。
    """

    if not class_votes:
        return None

    cls_id, votes = class_votes.most_common(1)[0]

    if votes < MIN_CLASS_VOTES_TO_LOCK:
        return None

    return cls_id