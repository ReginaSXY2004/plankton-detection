"""
视频可视化绘制模块

本文件负责：
1. 绘制 confirmed track 的检测框
2. 绘制 track id、类别名、置信度
3. 绘制各类别实时计数面板

本模块只负责“画到视频帧上”，不负责检测、跟踪或计数逻辑。
"""

import cv2

from src.config.infer_config import CLASS_NAMES, CLASS_COLORS


def draw_confirmed_track(frame, tr, rec):
    """
    在视频帧上绘制已确认的 track。

    参数：
    frame: 当前视频帧
    tr: Track 对象
    rec: 当前 track 对应的记录字典
    """

    x1, y1, x2, y2 = map(int, tr.bbox)

    cls_id = (
        rec["final_cls_id"]
        if rec["final_cls_id"] is not None
        else rec["last_cls_id"]
    )

    cls_name = (
        CLASS_NAMES.get(cls_id, f"cls{cls_id}")
        if cls_id is not None
        else "unknown"
    )

    color = CLASS_COLORS.get(cls_id, (0, 255, 0))

    cv2.rectangle(
        frame,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    text = f"id:{rec['display_id']} {cls_name} {tr.conf:.2f}"

    cv2.putText(
        frame,
        text,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA
    )


def draw_class_count_panel(frame, class_counts, total_count):
    """
    在视频左上角绘制实时分类计数面板。

    参数：
    frame: 当前视频帧
    class_counts: Counter，记录每个类别的实时计数
    total_count: 当前实时总计数
    """

    x0, y0 = 10, 80
    line_h = 24

    cv2.putText(
        frame,
        f"total:{total_count}",
        (x0, y0),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 200, 255),
        2,
        cv2.LINE_AA
    )

    sorted_items = sorted(
        class_counts.items(),
        key=lambda kv: kv[0]
    )

    for i, (cls_id, cnt) in enumerate(sorted_items, start=1):
        cls_name = CLASS_NAMES.get(cls_id, f"cls{cls_id}")
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))

        cv2.putText(
            frame,
            f"{cls_name}:{cnt}",
            (x0, y0 + i * line_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA
        )