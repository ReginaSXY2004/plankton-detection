"""
微生物检测与跟踪主入口

本文件负责：
1. 视频读取
2. YOLO 推理
3. detection 后处理
4. tracker 更新
5. 实时计数
6. best crop 管理
7. 视频与 CSV 输出
"""

from pathlib import Path
import csv
import cv2
import time
import shutil

from datetime import datetime, timedelta

from collections import Counter

from src.config.infer_config import *

from src.inference.detector import YoloDetector

from src.output.video_writer import VideoWriterWrapper

from src.output.async_video_writer import AsyncVideoWriter

from src.tracking.microbe_tracker import MicrobeTracker

from src.inference.postprocess import (
    deduplicate_detections,
    filter_blob_like_detections,
    is_duplicate_track_candidate,
)

from src.inference.crop_utils import (
    compute_sharpness,
    safe_crop,
    check_best_candidate,
    best_score,
    seconds_to_frames,
    majority_class_from_votes,
)

from src.output.visualization import (
    draw_confirmed_track,
    draw_class_count_panel,
)

def maybe_save_best_crop(rec, best_crop_dir: Path, src_fps: float):
    rec["save_fail_reason"] = ""

    if not rec["counted"]:
        rec["save_fail_reason"] = "not_counted"
        return False

    if rec["best_crop"] is None:
        rec["save_fail_reason"] = rec.get("last_best_update_status") or "no_valid_best_crop"
        return False

    show_id = rec["display_id"] if rec["display_id"] is not None else rec["track_id"]
    cls_name = CLASS_NAMES.get(rec["final_cls_id"], "unknown")

    best_timestamp_sec = rec["best_frame"] / max(src_fps, 1e-6)
    best_segment_id = rec.get("best_segment_id", -1)

    out_path = best_crop_dir / (
        f"{cls_name}"
        f"_seg_{best_segment_id:03d}"
        f"_t_{best_timestamp_sec:.1f}s"
        f"_showid_{show_id:03d}"
        f"_track_{rec['track_id']:03d}"
        f"_frame_{rec['best_frame']:05d}.png"
    )

    ok = cv2.imwrite(str(out_path), rec["best_crop"])
    rec["best_crop_path"] = str(out_path) if ok else ""
    rec["save_fail_reason"] = "saved" if ok else "imwrite_failed"
    return ok


def finalize_track_record(
    rec,
    confirmed_writer,
    best_crop_dir: Path,
    session_name: str,
    src_fps: float,
):
    if rec["finalized"]:
        return

    rec["finalized"] = True
    rec["saved"] = maybe_save_best_crop(rec, best_crop_dir, src_fps)

    first_timestamp_sec = rec["first_frame"] / max(src_fps, 1e-6)
    last_timestamp_sec = rec["last_frame"] / max(src_fps, 1e-6)
    best_timestamp_sec = rec["best_frame"] / max(src_fps, 1e-6) if rec["best_frame"] >= 0 else -1

    confirmed_writer.writerow([
        session_name,
        rec.get("first_segment_id", -1),
        rec.get("last_segment_id", -1),
        rec.get("best_segment_id", -1),
        
        rec.get("first_video_filename", ""),
        rec.get("last_video_filename", ""),
        rec.get("best_video_filename", ""),

        round(first_timestamp_sec, 3),
        round(last_timestamp_sec, 3),
        round(best_timestamp_sec, 3),

        rec["display_id"],
        rec["track_id"],
        rec["counted"],
        rec["saved"],
        rec["final_cls_id"],
        CLASS_NAMES.get(rec["final_cls_id"], "unknown") if rec["final_cls_id"] is not None else "unknown",
        rec["first_frame"],
        rec["last_frame"],
        rec["best_frame"],
        round(rec["best_conf"], 4),
        round(rec["best_sharpness"], 2),
        rec["best_w"],
        rec["best_h"],
        dict(rec["class_votes"]),
        rec.get("best_crop_path", ""),
        rec["save_fail_reason"],
        rec["last_best_update_status"],
    ])


def cleanup_track_records(
    track_records,
    counted_tracks,
    active_track_ids,
    current_frame_idx,
    stale_unconfirmed_frames,
    stale_duplicate_frames,
    stale_counted_frames,
):
    """
    长时间运行清理函数：
    1. finalized 且不 active 的 record
    2. 未 counted 且不 active 很久的 record
    3. duplicate 且不 active 的 record
    4. counted_tracks 中不 active 很久的旧 track
    """

    removed_records = 0
    removed_counted = 0
    released_best_crops = 0

    for tid in list(track_records.keys()):
        rec = track_records.get(tid)
        if rec is None:
            continue

        is_active = tid in active_track_ids
        inactive_frames = current_frame_idx - rec.get("last_frame", current_frame_idx)

        should_remove = False

        if (not is_active) and rec.get("finalized", False):
            should_remove = True

        elif (not is_active) and (not rec.get("counted", False)) and inactive_frames >= stale_unconfirmed_frames:
            should_remove = True

        elif (not is_active) and rec.get("is_duplicate", False) and inactive_frames >= stale_duplicate_frames:
            should_remove = True

        if should_remove:
            if rec.get("best_crop") is not None:
                rec["best_crop"] = None
                released_best_crops += 1

            track_records.pop(tid, None)
            counted_tracks.pop(tid, None)
            removed_records += 1

    for tid in list(counted_tracks.keys()):
        if tid in active_track_ids:
            continue

        rec = track_records.get(tid)
        if rec is None:
            counted_tracks.pop(tid, None)
            removed_counted += 1
            continue

        inactive_frames = current_frame_idx - rec.get("last_frame", current_frame_idx)

        if inactive_frames >= stale_counted_frames:
            counted_tracks.pop(tid, None)
            removed_counted += 1

    return {
        "removed_records": removed_records,
        "removed_counted": removed_counted,
        "released_best_crops": released_best_crops,
    }


def main():
    start_time = time.time()
    frame_count = 0
    last_print_time = start_time
    last_print_frame = 0
    cfg = get_infer_config(MAGNIFICATION)

    CONF = cfg["conf"]
    IMGSZ = cfg["imgsz"]
    DEDUP_IOU = cfg["dedup_iou"]
    DEDUP_CENTER = cfg["dedup_center"]

    print("=" * 60)
    print(f"MAGNIFICATION: {MAGNIFICATION}x")
    print(f"CONF: {CONF}")
    print(f"IMGSZ: {IMGSZ}")
    print(f"DEDUP_IOU: {DEDUP_IOU}")
    print(f"DEDUP_CENTER: {DEDUP_CENTER}")
    print(f"TRACKER_CONFIG: {cfg['tracker']}")
    print(f"MODEL_PATH: {MODEL_PATH}")
    print("=" * 60)

    detector = YoloDetector(MODEL_PATH, device=DEVICE)

    # 一个 session 对应一次连续运行任务。
    # 对真实部署来说，对应一次机器人下水/一次连续检测。
    session_start_dt = datetime.now()
    session_name = session_start_dt.strftime("%Y-%m-%d_T%H-%M-%S")

    track_analysis_dir = (
        PROJECT_ROOT
        / "runs"
        / "track_analysis"
        / f"{VIDEO_STEM}_{session_name}"
    )

    # 当前仍是实验脚本逻辑：每次运行清空同名 session 目录。
    # 由于 session_name 带时间戳，正常不会误删历史结果。
    if track_analysis_dir.exists():
        shutil.rmtree(track_analysis_dir)

    video_dir = track_analysis_dir / "videos"
    csv_dir = track_analysis_dir / "csv"
    best_crop_dir = track_analysis_dir / "best_crops"
    log_dir = track_analysis_dir / "logs"

    video_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)
    best_crop_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    confirmed_csv_path = csv_dir / "confirmed_microbes.csv"
    debug_csv_path = csv_dir / "confirmed_tracks_debug.csv"

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {VIDEO_PATH}")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0

    confirm_min_hits = seconds_to_frames(
    CONFIRM_SECONDS,
    src_fps,
    min_frames=3
    )

    finalize_missed_thresh = seconds_to_frames(
        FINALIZE_MISSED_SECONDS,
        src_fps,
        min_frames=3
    )

    max_missing_seconds = MAX_MISSING_SECONDS_BY_MAG.get(float(MAGNIFICATION), 0.60)
    max_missing_frames = seconds_to_frames(
        max_missing_seconds,
        src_fps,
        min_frames=3
    )

    reconnect_max_missing = seconds_to_frames(
        RECONNECT_SECONDS,
        src_fps,
        min_frames=2
    )

    stale_unconfirmed_frames = seconds_to_frames(
        2.0,
        src_fps,
        min_frames=max_missing_frames + 1
    )

    stale_duplicate_frames = seconds_to_frames(
        1.0,
        src_fps,
        min_frames=max_missing_frames + 1
    )

    stale_counted_frames = seconds_to_frames(
        2.0,
        src_fps,
        min_frames=max_missing_frames + 1
    )

    cfg["tracker"]["max_missing"] = max_missing_frames
    cfg["tracker"]["reconnect_max_missing"] = reconnect_max_missing

    print(f"src_fps: {src_fps}")
    print(f"confirm_min_hits: {confirm_min_hits} frames ({CONFIRM_SECONDS}s)")
    print(f"finalize_missed_thresh: {finalize_missed_thresh} frames ({FINALIZE_MISSED_SECONDS}s)")
    print(f"max_missing_frames: {max_missing_frames} frames ({max_missing_seconds}s)")
    print(f"reconnect_max_missing: {reconnect_max_missing} frames ({RECONNECT_SECONDS}s)")
    print(f"stale_unconfirmed_frames: {stale_unconfirmed_frames}")
    print(f"stale_duplicate_frames: {stale_duplicate_frames}")
    print(f"stale_counted_frames: {stale_counted_frames}")


    tracker = MicrobeTracker(**cfg["tracker"])

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    segment_id = 0
    segment_frames = None
    current_video_path = None

    if SAVE_VIDEO_EVERY_N_FRAMES > 0:
        output_fps = src_fps / SAVE_VIDEO_EVERY_N_FRAMES

        # segment_frames 表示多少“原始输入帧”后切一个新视频。
        # 注意：这里按原始 frame_idx 切分，而不是按实际写入帧数切分。
        segment_frames = int(src_fps * VIDEO_SEGMENT_MINUTES * 60)

        segment_start_dt = session_start_dt
        segment_time_str = segment_start_dt.strftime("%H-%M-%S")
        current_video_path = video_dir / f"video_{segment_id:03d}_{segment_time_str}.mp4"

        base_writer = VideoWriterWrapper(
            out_video_path=current_video_path,
            width=width,
            height=height,
            fps=output_fps,
            backend=VIDEO_WRITER_BACKEND,
            bitrate=VIDEO_BITRATE,
        )

        writer = AsyncVideoWriter(
            base_writer,
            max_queue_size=16,
        )

        print(f"当前视频 segment：{current_video_path}")

    # internal track_id -> record
    track_records = {}
    counted_tracks = {}
    realtime_count = 0
    class_counts = Counter()

    f_debug = None
    debug_writer = None

    try:
        if SAVE_DEBUG_CSV:
            f_debug = open(debug_csv_path, "w", newline="", encoding="utf-8")
            debug_writer = csv.writer(f_debug)
            debug_writer.writerow([
                "frame_idx", "display_id", "track_id", "cls_id", "cls_name", "conf",
                "x1", "y1", "x2", "y2", "cx", "cy", "w", "h", "hits", "missed"
            ])

        with open(confirmed_csv_path, "w", newline="", encoding="utf-8") as f_confirmed:
            confirmed_writer = csv.writer(f_confirmed)
            confirmed_writer.writerow([
                "session_name",
                "first_segment_id",
                "last_segment_id",
                "best_segment_id",
                "first_video_filename",
                "last_video_filename",
                "best_video_filename",

                "first_timestamp_sec",
                "last_timestamp_sec",
                "best_timestamp_sec",

                "display_id",
                "track_id",
                "counted",
                "saved",
                "final_cls_id",
                "final_cls_name",
                "first_frame",
                "last_frame",
                "best_frame",
                "best_conf",
                "best_sharpness",
                "best_w",
                "best_h",
                "class_votes",
                "best_crop_path",
                "save_fail_reason",
                "last_best_update_status",
            ])

            frame_idx = 0
            while True:
                t0 = time.time()
                ret, frame = cap.read()
                if not ret:
                    break
                
                # YOLO 检测：返回统一的 Detection list
                detections = detector.detect(
                    frame,
                    conf=CONF,
                    imgsz=IMGSZ
                )
                
                t1 = time.time()

                raw_det_count = len(detections)

                # 去重，过滤圆形亮斑
                detections = deduplicate_detections(
                    detections,
                    iou_thresh=DEDUP_IOU,
                    center_thresh=DEDUP_CENTER
                )

                detections = filter_blob_like_detections(
                    frame,
                    detections,
                    circularity_thresh=0.82,
                    texture_std_thresh=18.0,
                    min_box_size=10
                )

                t2 = time.time()

                kept_det_count = len(detections)

                tracks = tracker.update(detections)

                t3 = time.time()

                tracks_to_draw = []


                active_track_ids = set(tracker.tracks.keys())

                cleanup_stats = cleanup_track_records(
                    track_records=track_records,
                    counted_tracks=counted_tracks,
                    active_track_ids=active_track_ids,
                    current_frame_idx=frame_idx,
                    stale_unconfirmed_frames=stale_unconfirmed_frames,
                    stale_duplicate_frames=stale_duplicate_frames,
                    stale_counted_frames=stale_counted_frames,
                )

                for tr in tracks:
                    tid = tr.track_id

                    if tid not in track_records:
                        track_records[tid] = {
                            
                            "track_id": tid,
                            "display_id": None,
                            "first_frame": frame_idx,
                            "last_frame": frame_idx,
                            "best_frame": -1,
                            "best_conf": 0.0,
                            "best_sharpness": 0.0,
                            "best_w": 0,
                            "best_h": 0,
                            "best_crop": None,
                            "best_score": -1e9,
                            "counted": False,
                            "saved": False,
                            "finalized": False,
                            "is_duplicate": False,
                            "duplicate_of": None,
                            "class_votes": Counter(),
                            "final_cls_id": None,
                            "last_cls_id": None,
                            "save_fail_reason": "",
                            "last_best_update_status": "",
                            "first_segment_id": segment_id,
                            "last_segment_id": segment_id,
                            "best_segment_id": -1,
                            "first_video_filename": "" if current_video_path is None else current_video_path.name,
                            "last_video_filename": "" if current_video_path is None else current_video_path.name,
                            "best_video_filename": "",
                            "best_crop_path": "",
                        }

                    rec = track_records[tid]
                    rec["last_frame"] = frame_idx
                    rec["last_segment_id"] = segment_id
                    rec["last_video_filename"] = (
                        "" if current_video_path is None
                        else current_video_path.name
                    )
                    rec["last_cls_id"] = tr.cls_id
                    rec["class_votes"][tr.cls_id] += 1

                    # 更新当前投票主类
                    rec["final_cls_id"] = majority_class_from_votes(rec["class_votes"])

                    # 裁当前帧的原图区域，还没画框
                    crop, _ = safe_crop(frame, tr.bbox)
                    sharpness = compute_sharpness(crop) if crop is not None else 0.0
                    area = float((tr.x2 - tr.x1) * (tr.y2 - tr.y1))

                    # 判断截图有没有资格竞争 best crop
                    ok_best, reason = check_best_candidate(tr, crop, sharpness)

                    if not ok_best:
                        rec["last_best_update_status"] = reason
                    else:
                        # 如果合格，就和历史 best 比分
                        score = best_score(sharpness, tr.conf, area)
                        if score > rec["best_score"]:
                            rec["best_score"] = score
                            rec["best_frame"] = frame_idx
                            rec["best_conf"] = tr.conf
                            rec["best_sharpness"] = sharpness
                            rec["best_w"] = 0 if crop is None else crop.shape[1]
                            rec["best_h"] = 0 if crop is None else crop.shape[0]
                            rec["best_crop"] = None if crop is None else crop.copy() # best crop 每个 track 只保留目前最好的那一张
                            rec["best_segment_id"] = segment_id
                            rec["best_video_filename"] = (
                                "" if current_video_path is None
                                else current_video_path.name
                            )
                            rec["last_best_update_status"] = "accepted_as_best"

                    # track 达到条件之后才counted
                    if (not rec["counted"]) and (not rec["is_duplicate"]) and tr.hits >= confirm_min_hits:
                        duplicate_of = None

                        for old_tid, old_tr in counted_tracks.items():
                            if is_duplicate_track_candidate(tr, old_tr):
                                duplicate_of = old_tid
                                break

                        if duplicate_of is not None:
                            rec["is_duplicate"] = True
                            rec["duplicate_of"] = duplicate_of
                            rec["counted"] = False
                            rec["display_id"] = track_records[duplicate_of]["display_id"]
                        else:
                            realtime_count += 1
                            rec["counted"] = True
                            rec["display_id"] = realtime_count

                            locked_cls = rec["final_cls_id"] if rec["final_cls_id"] is not None else tr.cls_id
                            class_counts[locked_cls] += 1

                            counted_tracks[tid] = tr

                    if rec["counted"] and (not rec["is_duplicate"]):
                        tracks_to_draw.append((tr, rec))

                        if SAVE_DEBUG_CSV and debug_writer is not None:
                            x1, y1, x2, y2 = tr.bbox
                            w = x2 - x1
                            h = y2 - y1
                            cls_id = rec["final_cls_id"] if rec["final_cls_id"] is not None else tr.cls_id
                            debug_writer.writerow([
                                frame_idx,
                                rec["display_id"],
                                tr.track_id,
                                cls_id,
                                CLASS_NAMES.get(cls_id, "unknown"),
                                round(tr.conf, 4),
                                round(x1, 2),
                                round(y1, 2),
                                round(x2, 2),
                                round(y2, 2),
                                round(tr.cx, 2),
                                round(tr.cy, 2),
                                round(w, 2),
                                round(h, 2),
                                tr.hits,
                                tr.missed
                            ])

                # 统一画框
                for tr, rec in tracks_to_draw:
                    draw_confirmed_track(frame, tr, rec)


                for tid, tr in list(tracker.tracks.items()):
                    rec = track_records.get(tid)
                    if rec is None:
                        continue

                    if rec["finalized"]:
                        continue

                    if tr.missed >= finalize_missed_thresh and rec["counted"]:
                        finalize_track_record(
                            rec,
                            confirmed_writer,
                            best_crop_dir,
                            session_name,
                            src_fps,
                        )
                        # 释放最占内存的图片，但保留 record 防止重复计数
                        rec["best_crop"] = None


                cv2.putText(
                    frame,
                    f"frame:{frame_idx}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA
                )

                if SHOW_CLASS_COUNTS_ON_VIDEO:
                    draw_class_count_panel(frame, class_counts, realtime_count)

                t4 = time.time()

                # 长时间运行时，按固定时长切分视频 segment。
                # 短视频不足 VIDEO_SEGMENT_MINUTES 时不会触发切分，仍然正常保存为 video_000_xxx.mp4。
                if (
                    writer is not None
                    and segment_frames is not None
                    and frame_idx > 0
                    and frame_idx % segment_frames == 0
                ):
                    segment_id += 1

                    segment_start_dt = session_start_dt + timedelta(
                        seconds=frame_idx / max(src_fps, 1e-6)
                    )
                    segment_time_str = segment_start_dt.strftime("%H-%M-%S")
                    current_video_path = video_dir / f"video_{segment_id:03d}_{segment_time_str}.mp4"

                    print(f"[Video Segment] 切换到新视频：{current_video_path}")
                    writer.reopen(current_video_path)

                # 视频写入在所有可视化绘制完成后执行，
                # 因此输出视频包含 bbox、track id、frame id 和实时计数面板。
                if (
                    writer is not None
                    and frame_idx % SAVE_VIDEO_EVERY_N_FRAMES == 0
                ):
                    writer.write(frame)

                t5 = time.time()

                frame_idx += 1
                frame_count += 1

                if PRINT_FPS and frame_count % 120 == 0:
                    now = time.time()

                    total_elapsed = now - start_time
                    avg_fps = frame_count / max(total_elapsed, 1e-6)

                    window_elapsed = now - last_print_time
                    window_frames = frame_count - last_print_frame
                    window_fps = window_frames / max(window_elapsed, 1e-6)
                    writer_stats = (
                        writer.stats()
                        if writer is not None and hasattr(writer, "stats")
                        else {
                            "qsize": 0,
                            "dropped_frames": 0,
                            "written_frames": 0,
                        }
                    )

                    print(
                        f"[FPS] window={window_fps:.2f} avg={avg_fps:.2f} | "
                        f"raw_det={raw_det_count} kept_det={kept_det_count} | "
                        f"active_tracks={len(tracker.tracks)} "
                        f"visible_tracks={len(tracks_to_draw)} "
                        f"track_records={len(track_records)} | "
                        f"writer_q={writer_stats['qsize']} "
                        f"writer_drop={writer_stats['dropped_frames']} "
                        f"writer_written={writer_stats['written_frames']} | "
                        f"cleanup_rec={cleanup_stats['removed_records']} "
                        f"cleanup_counted={cleanup_stats['removed_counted']} "
                        f"released_crop={cleanup_stats['released_best_crops']} "
                    )

                    print(
                        f"[TIME] "
                        f"yolo={(t1-t0)*1000:.1f}ms "
                        f"post={(t2-t1)*1000:.1f}ms "
                        f"track={(t3-t2)*1000:.1f}ms "
                        f"draw={(t4-t3)*1000:.1f}ms "
                        f"write={(t5-t4)*1000:.1f}ms"
                    )
                    last_print_time = now
                    last_print_frame = frame_count

            for tid, rec in list(track_records.items()):
                if rec["counted"] and (not rec["finalized"]):
                    finalize_track_record(
                        rec,
                        confirmed_writer,
                        best_crop_dir,
                        session_name,
                        src_fps,
                    )

                if rec["finalized"]:
                    track_records.pop(tid, None)
                    counted_tracks.pop(tid, None)

            total_time = time.time() - start_time
            avg_fps = frame_count / max(total_time, 1e-6)
            print(f"\n平均FPS: {avg_fps:.2f}")

    finally:
        cap.release()
        if writer is not None:
            writer.release()

        if f_debug is not None:
            f_debug.close()

    if SAVE_VIDEO_EVERY_N_FRAMES > 0:
        print(f"完成视频目录：{video_dir}")
    else:
        print("完成视频：未保存（SAVE_VIDEO_EVERY_N_FRAMES=0）")

    if SAVE_DEBUG_CSV:
        print(f"confirmed 逐帧 debug CSV：{debug_csv_path}")
    else:
        print("confirmed 逐帧 debug CSV：未保存（SAVE_DEBUG_CSV=False）")

    print(f"confirmed 汇总 CSV：{confirmed_csv_path}")
    print(f"最佳图目录：{best_crop_dir}")
    print(f"实时总计数：{realtime_count}")

    print("分类别计数：")
    for cls_id in sorted(class_counts.keys()):
        print(f"  {CLASS_NAMES.get(cls_id, cls_id)}: {class_counts[cls_id]}")

    print("\n===== Runtime Monitor Description =====")
    print("window          : 最近窗口 FPS（瞬时速度）")
    print("avg             : 从开始到现在的平均 FPS")
    print("raw_det         : YOLO 原始 detection 数量")
    print("kept_det        : 后处理后保留的 detection 数量")
    print("active_tracks   : 当前 tracker 中的轨迹总数")
    print("visible_tracks  : 当前已 confirmed 并显示的轨迹数")
    print("track_records   : 当前内存中的历史 track record 数量")
    print("writer_q        : 当前 writer queue 中等待写入的帧数")
    print("writer_drop     : 因 queue 满而被丢弃的 debug 视频帧数")
    print("writer_written  : 后台线程已成功写入的视频帧数")
    print("=======================================")


if __name__ == "__main__":
    main()