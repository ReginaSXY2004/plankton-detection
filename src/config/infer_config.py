"""
微生物视频推理配置文件

本文件负责：
1. 模型与视频路径
2. 类别名与可视化颜色
3. 推理运行开关
4. best crop 筛选参数
5. tracker 时间控制参数
6. 不同倍率下的推理配置

"""

from pathlib import Path


# =========================================================
# Project Root
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =========================================================
# Model / Video Paths
# =========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "runs"
    / "yolov8n_1x_multiclass_v1"
    / "weights"
    / "best.pt"
)

VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "video1x"
    / "Sample17.avi"
)

VIDEO_STEM = VIDEO_PATH.stem

TRACK_ANALYSIS_DIR = (
    PROJECT_ROOT
    / "runs"
    / "track_analysis"
    / VIDEO_STEM
)

OUT_VIDEO = (
    TRACK_ANALYSIS_DIR
    / "output_video.mp4"
)

OUT_DEBUG_CSV = (
    TRACK_ANALYSIS_DIR
    / "confirmed_tracks_debug.csv"
)

OUT_CONFIRMED_CSV = (
    TRACK_ANALYSIS_DIR
    / "confirmed_microbes.csv"
)

BEST_CROP_DIR = (
    TRACK_ANALYSIS_DIR
    / "best_crops"
)


# =========================================================
# Device
# =========================================================

DEVICE = 0


# =========================================================
# Current Magnification
# =========================================================

MAGNIFICATION = 1.0


# =========================================================
# Class Names
# =========================================================

CLASS_NAMES = {
    0: "daxingzao",
    1: "jianshuizao",
    2: "xiannvchong",
    3: "lunchong",
    4: "xiangbizao",
    5: "weizhi",
    6: "xianchong",
}


# =========================================================
# Visualization Colors
# =========================================================

CLASS_COLORS = {
    0: (255, 80, 80),
    1: (80, 255, 255),
    2: (80, 255, 80),
    3: (80, 80, 255),
    4: (180, 80, 255),
    5: (255, 80, 200),
    6: (255, 180, 80),
}


# =========================================================
# Runtime Switches
# =========================================================

# 0 = disable saving
# 1 = save every frame
# 2 = save every 2 frames
# 3 = save every 3 frames

SAVE_VIDEO_EVERY_N_FRAMES = 1

SAVE_DEBUG_CSV = False

PRINT_FPS = True

SHOW_CLASS_COUNTS_ON_VIDEO = True


# =========================================================
# Video Writer Backend
# =========================================================

# opencv_mp4v
# ffmpeg_nvenc
# jetson_gstreamer
# none

VIDEO_WRITER_BACKEND = "ffmpeg_nvenc"

VIDEO_BITRATE = "8M"

# =========================================================
# Long-Running Session / Video Segmentation
# =========================================================

# 长时间部署时，单个 mp4 文件过大且异常退出风险更高。
# 因此系统按固定时长切分视频 segment。
VIDEO_SEGMENT_MINUTES = 30


# =========================================================
# Best Crop Thresholds
# =========================================================

BEST_MIN_W = 12
BEST_MIN_H = 12

BEST_MIN_CONF = 0.18

BEST_MIN_SHARPNESS = 6.0


# =========================================================
# Time-Based Tracking Controls
# =========================================================

CONFIRM_SECONDS = 0.20

FINALIZE_MISSED_SECONDS = 0.30

RECONNECT_SECONDS = 0.5


# =========================================================
# Max Missing Time by Magnification
# =========================================================

MAX_MISSING_SECONDS_BY_MAG = {
    1.0: 0.60,
    0.5: 0.70,
    0.2: 0.50,
    2.0: 0.35,
}


# =========================================================
# Class Voting
# =========================================================

MIN_CLASS_VOTES_TO_LOCK = 3


# =========================================================
# Magnification-Specific Config
# =========================================================

def get_infer_config(magnification: float):

    config = {
        "conf": 0.35,
        "imgsz": 800,
        "dedup_iou": 0.62,
        "dedup_center": 16,

        "tracker": dict(
            min_hits_to_show=6,
            base_distance_thresh=18.0,
            distance_scale=1.4,
            max_size_ratio=2.0,
            conf_threshold_for_tracking=0.22,
            no_spawn_radius=24.0,
            debug_print=False,
        )
    }

    if magnification == 1.0:
        return config

    elif magnification == 0.5:
        return {
            "conf": 0.35,
            "imgsz": 800,
            "dedup_iou": 0.60,
            "dedup_center": 12,

            "tracker": dict(
                min_hits_to_show=6,
                base_distance_thresh=16.0,
                distance_scale=1.5,
                max_size_ratio=2.0,
                conf_threshold_for_tracking=0.18,
                no_spawn_radius=20.0,
                debug_print=False,
            )
        }

    elif magnification == 0.2:
        return {
            "conf": 0.40,
            "imgsz": 800,
            "dedup_iou": 0.55,
            "dedup_center": 8,

            "tracker": dict(
                min_hits_to_show=4,
                base_distance_thresh=8.0,
                distance_scale=1.2,
                max_size_ratio=2.0,
                conf_threshold_for_tracking=0.12,
                no_spawn_radius=16.0,
                debug_print=False,
            )
        }

    elif magnification == 2.0:
        return {
            "conf": 0.40,
            "imgsz": 640,
            "dedup_iou": 0.70,
            "dedup_center": 28,

            "tracker": dict(
                min_hits_to_show=3,
                base_distance_thresh=28.0,
                distance_scale=1.5,
                max_size_ratio=2.2,
                conf_threshold_for_tracking=0.28,
                no_spawn_radius=50.0,
                debug_print=False,
            )
        }

    print(
        f"[warning] Undefined magnification {magnification}x. "
        f"Fallback to default 1.0x config."
    )

    return config