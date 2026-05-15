"""
test_writer_speed.py

用途：
    单独测试视频写入链路（writer pipeline）的极限吞吐。

特点：
    - 不运行 YOLO
    - 不运行 tracking
    - 不画框
    - 只执行：
        cap.read() -> writer.write(frame)

目的：
    用于判断性能瓶颈是否来自：
        1. TensorRT / YOLO 推理
        2. tracking / 后处理
        3. 视频写入 pipeline（GStreamer / FFmpeg / NVENC）

典型使用场景：
    - Jetson 边缘部署性能分析
    - Debug writer_drop 持续增长问题
    - 对比不同 writer backend 的吞吐
    - 测试不同 bitrate / resolution / pipeline 配置

观察重点：
    writer_q:
        writer queue 当前积压帧数

    writer_drop:
        因 queue 满而被丢弃的帧数

    writer_written:
        后台线程实际成功写入的视频帧数

结论解释：
    如果不运行 YOLO 时仍然大量 drop，
    则说明瓶颈主要位于 writer pipeline，
    而不是检测主流程。
"""


import time
import cv2

from pathlib import Path

from src.config.infer_config import (
    VIDEO_PATH,
    VIDEO_WRITER_BACKEND,
    VIDEO_BITRATE,
    PROJECT_ROOT,
)

from src.output.video_writer import VideoWriterWrapper
from src.output.async_video_writer import AsyncVideoWriter


def main():

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {VIDEO_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = (
        PROJECT_ROOT
        / "runs"
        / "writer_test.mp4"
    )

    base_writer = VideoWriterWrapper(
        out_video_path=out_path,
        width=width,
        height=height,
        fps=src_fps,
        backend=VIDEO_WRITER_BACKEND,
        bitrate=VIDEO_BITRATE,
    )

    writer = AsyncVideoWriter(
        base_writer,
        max_queue_size=64,
    )

    frame_count = 0
    start_time = time.time()

    last_print = start_time
    last_frame = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        # 不做 YOLO
        # 不做 tracking
        # 不画框
        # 纯测试 writer 极限吞吐

        writer.write(frame)

        frame_count += 1

        if frame_count % 120 == 0:

            now = time.time()

            elapsed = now - start_time
            avg_fps = frame_count / max(elapsed, 1e-6)

            window_elapsed = now - last_print
            window_frames = frame_count - last_frame
            window_fps = window_frames / max(window_elapsed, 1e-6)

            stats = writer.stats()

            print(
                f"[WriterTest] "
                f"window={window_fps:.2f} "
                f"avg={avg_fps:.2f} | "
                f"q={stats['qsize']} "
                f"drop={stats['dropped_frames']} "
                f"written={stats['written_frames']}"
            )

            last_print = now
            last_frame = frame_count

    cap.release()

    writer.release()

    total_time = time.time() - start_time

    print(f"\n平均FPS: {frame_count / total_time:.2f}")
    print(f"输出视频: {out_path}")


if __name__ == "__main__":
    main()