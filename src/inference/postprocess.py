"""
检测后处理模块（postprocess）

本文件负责：
1. detection 去重
2. IoU 计算
3. 圆形亮斑过滤
4. 重复 track 候选判断

目的：
将 YOLO 输出结果进一步清洗，
减少重复框、亮斑误检与重复计数问题。
"""

import math
import cv2
import numpy as np


def box_iou_xyxy(a, b):
    """
    计算两个 xyxy bbox 的 IoU
    """

    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)

    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter + 1e-6

    return inter / union


def point_in_box(cx, cy, box):
    """
    判断点是否位于 bbox 内
    """

    x1, y1, x2, y2 = box

    return x1 <= cx <= x2 and y1 <= cy <= y2


def compute_circularity_from_roi(frame, det):
    """
    从 ROI 中计算圆形度与纹理复杂度

    用于过滤：
    高圆形度 + 低纹理 的亮斑噪声
    """

    x1, y1, x2, y2 = map(
        int,
        [det.x1, det.y1, det.x2, det.y2]
    )

    h, w = frame.shape[:2]

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(0, min(x2, w))
    y2 = max(0, min(y2, h))

    if x2 <= x1 or y2 <= y1:
        return None, None

    roi = frame[y1:y2, x1:x2]

    if roi.size == 0:
        return None, None

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    _, binary = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, float(np.std(gray))

    cnt = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if perimeter <= 1e-6:
        return None, float(np.std(gray))

    circularity = (
        4.0 * math.pi * area /
        (perimeter * perimeter)
    )

    texture_std = float(np.std(gray))

    return circularity, texture_std


def filter_blob_like_detections(
    frame,
    detections,
    circularity_thresh=0.82,
    texture_std_thresh=18.0,
    min_box_size=10
):
    """
    过滤高圆形度亮斑

    这些目标通常：
    1. 形状接近圆
    2. 内部纹理变化较少
    """

    kept = []

    for det in detections:

        bw = det.x2 - det.x1
        bh = det.y2 - det.y1

        if bw < min_box_size or bh < min_box_size:
            kept.append(det)
            continue

        circularity, texture_std = (
            compute_circularity_from_roi(frame, det)
        )

        if circularity is None:
            kept.append(det)
            continue

        if (
            circularity > circularity_thresh
            and texture_std < texture_std_thresh
        ):
            continue

        kept.append(det)

    return kept


def deduplicate_detections(
    detections,
    iou_thresh=0.65,
    center_thresh=18
):
    """
    对 detection 去重

    避免：
    同一微生物被 YOLO 输出多个重叠框
    """

    detections = sorted(
        detections,
        key=lambda d: d.conf,
        reverse=True
    )

    kept = []

    for det in detections:

        keep = True

        for k in kept:

            iou = box_iou_xyxy(
                (det.x1, det.y1, det.x2, det.y2),
                (k.x1, k.y1, k.x2, k.y2)
            )

            center_dist = (
                (
                    (det.cx - k.cx) ** 2
                    + (det.cy - k.cy) ** 2
                ) ** 0.5
            )

            if (
                iou > iou_thresh
                or center_dist < center_thresh
            ):
                keep = False
                break

        if keep:
            kept.append(det)

    return kept


def is_duplicate_track_candidate(new_tr, old_tr) -> bool:
    """
    判断两个 track 是否可能是同一个微生物

    用于 counted 前的二次去重，
    避免同一目标重新生成新 ID。
    """

    new_box = new_tr.bbox
    old_box = old_tr.bbox

    iou = box_iou_xyxy(new_box, old_box)

    center_dist = (
        (
            (new_tr.cx - old_tr.cx) ** 2
            + (new_tr.cy - old_tr.cy) ** 2
        ) ** 0.5
    )

    mutual_center_inside = (
        point_in_box(
            new_tr.cx,
            new_tr.cy,
            old_box
        )
        and
        point_in_box(
            old_tr.cx,
            old_tr.cy,
            new_box
        )
    )

    dynamic_center_thresh = 0.35 * max(
        new_tr.w,
        new_tr.h,
        old_tr.w,
        old_tr.h
    )

    similar_size = (
        max(new_tr.w, old_tr.w)
        / max(min(new_tr.w, old_tr.w), 1e-6)
        < 1.8
        and
        max(new_tr.h, old_tr.h)
        / max(min(new_tr.h, old_tr.h), 1e-6)
        < 1.8
    )

    return (
        similar_size
        and (
            iou > 0.35
            or mutual_center_inside
            or center_dist < dynamic_center_thresh
        )
    )