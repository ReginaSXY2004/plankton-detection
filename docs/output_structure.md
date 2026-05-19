# 输出目录结构说明

本文档说明项目运行后的输出目录结构、CSV 文件、best crop 文件以及 runtime monitor 的含义。

---

## 1. 输出根目录

项目运行后，默认输出到：

```text
runs/track_analysis/
```

每次运行会创建一个新的 session 文件夹。

示例：

```text
runs/track_analysis/
└── Sample17_2026-05-13_T11-23-54/
```

session 文件夹名称通常由以下部分组成：

```text
视频名_运行日期_运行开始时间
```

---

## 2. 单次运行的目录结构

示例：

```text
Sample17_2026-05-13_T11-23-54/
├── videos/
├── csv/
├── best_crops/
└── logs/
```

说明：

- `videos/`：保存带检测框、track id、计数面板的输出视频
- `csv/`：保存 confirmed track 汇总 CSV 和可选 debug CSV（目前为关闭状态）
- `best_crops/`：保存每个 confirmed microbe 的最佳截图
- `logs/`：预留日志目录

当前项目的大部分调试、统计与长期运行状态，
均通过这些输出目录进行分析。

---

## 3. videos/

`videos/` 用于保存输出调试视频。

示例：

```text
videos/
├── video_000_11-23-54.mp4
├── video_001_11-53-54.mp4
└── video_002_12-23-54.mp4
```

长时间运行时，系统会按固定时长（比如30分钟）切分视频 segment。

这样做的原因：

- 避免单个视频文件过大
- 降低异常退出导致整段视频损坏的风险
- 方便后续按时间段回看
- 更适合长时间部署

输出视频中通常包含：

- bbox 检测框
- track id
- 类别名
- confidence
- 当前 frame id
- 实时总计数
- 分类别计数面板

---

## 4. csv/

`csv/` 用于保存统计结果。

示例：

```text
csv/
├── confirmed_microbes.csv
└── confirmed_tracks_debug.csv
```

---

## 5. confirmed_microbes.csv

`confirmed_microbes.csv` 是正式汇总 CSV。

每个 confirmed microbe 对应一行。

常见字段包括：

```text
session_name
first_segment_id
last_segment_id
best_segment_id
first_video_filename
last_video_filename
best_video_filename
first_timestamp_sec
last_timestamp_sec
best_timestamp_sec
display_id
track_id
counted
saved
final_cls_id
final_cls_name
first_frame
last_frame
best_frame
best_conf
best_sharpness
best_w
best_h
class_votes
best_crop_path
save_fail_reason
last_best_update_status
```

重点字段说明：

| 字段 | 含义 |
|---|---|
| `session_name` | 当前运行 session 名称 |
| `display_id` | 视频中显示给用户看的计数 ID |
| `track_id` | tracker 内部 ID |
| `counted` | 是否进入正式计数 |
| `saved` | 是否成功保存 best crop |
| `final_cls_name` | 最终类别名 |
| `first_frame` | 该 track 第一次出现的帧 |
| `last_frame` | 该 track 最后一次出现的帧 |
| `best_frame` | best crop 所在帧 |
| `best_conf` | best crop 对应 detection confidence |
| `best_sharpness` | best crop 清晰度 |
| `best_crop_path` | best crop 图片保存路径 |
| `save_fail_reason` | 保存成功或失败原因 |
| `last_best_update_status` | best crop 最后一次更新状态 |

---

## 6. confirmed_tracks_debug.csv

`confirmed_tracks_debug.csv` 是可选逐帧 debug CSV。

只有在配置中开启：

```python
SAVE_DEBUG_CSV = True
```

时才会生成。

它用于逐帧分析 tracking 行为，适合调试：

- track 是否稳定
- ID 是否断裂
- bbox 是否抖动
- 类别是否频繁变化
- missed / hits 是否异常

默认部署时通常可以关闭，避免 CSV 文件过大，运行缓慢。

---

## 7. best_crops/

`best_crops/` 用于保存每个 confirmed microbe 的代表性截图。

示例：

```text
best_crops/
├── xiangbizao_seg_000_t_12.3s_showid_015_track_023_frame_00321.png
├── jianshuizao_seg_000_t_18.7s_showid_016_track_024_frame_00498.png
└── lunchong_seg_001_t_902.1s_showid_088_track_117_frame_29987.png
```

文件名中包含：

| 部分 | 含义 |
|---|---|
| 类别名 | 该 track 的最终类别 |
| `seg_000` | best crop 所在视频 segment |
| `t_12.3s` | best crop 对应时间 |
| `showid_015` | 视频显示 ID |
| `track_023` | tracker 内部 ID |
| `frame_00321` | best crop 所在帧 |

文件名设计原因：
任何输出文件都应该能反向定位到：
视频时间
track
segment
frame

---

## 8. best crop 选择逻辑

系统只对每个 track 只保留当前最优的一张。（不是每个真实track都能成功保存best crop，比如sharpness不够会导致放弃保存）

当前评分综合考虑：

```text
sharpness
confidence
bbox area
```

优势：

- 减少磁盘占用
- 减少内存占用
- 保存更清晰、更有代表性的截图
- 方便后续人工检查

*best crop 评分策略可根据后续真实部署数据继续调整。

---

## 9. Track 生命周期

当前 track 生命周期：

```text
active track
→ missed
→ finalize
→ 写入 confirmed_microbes.csv
→ 保存 best crop
→ 释放 best_crop 内存
→ cleanup 删除 record
```

对于未 confirmed 的短命 track：

```text
短暂误检 / hits 不够
→ 不进入正式计数
→ 不写 confirmed_microbes.csv
→ cleanup 后从内存中删除
```

因此 cleanup 不会影响正式统计结果。

---

## 10. cleanup 负责什么

cleanup 会清理：

- finalized 且已经不 active 的 record
- 未 counted 且不 active 很久的短命 record
- duplicate 且不 active 的 record
- counted_tracks 中不 active 太久的旧对象
- 已无用的 best_crop 内存引用

目的：

- 防止长时间运行时 `track_records` 无限增长
- 防止短命误检积累
- 防止 best crop 长期占用内存
- 提高边缘端长时间部署稳定性

---

## 11. Runtime Monitor

运行时终端会周期性输出类似信息：

```text
[FPS] window=39.57 avg=38.74 |
raw_det=2 kept_det=1 |
active_tracks=2 visible_tracks=1 track_records=2 |
writer_q=1 writer_drop=0 writer_written=2399 |
cleanup_rec=1 cleanup_counted=0 released_crop=0
```

字段说明：

| 字段 | 含义 |
|---|---|
| `window` | 最近窗口 FPS，反映瞬时速度 |
| `avg` | 从程序开始到当前的平均 FPS |
| `raw_det` | YOLO 原始 detection 数量 |
| `kept_det` | 后处理后保留的 detection 数量 |
| `active_tracks` | tracker 当前维护的 track 数量 |
| `visible_tracks` | 当前 confirmed 且被绘制的 track 数量 |
| `track_records` | 当前内存中的 track record 数量 |
| `writer_q` | 异步视频 writer 队列长度 |
| `writer_drop` | writer 队列满时丢弃的视频帧数 |
| `writer_written` | 后台线程已写入的视频帧数 |
| `cleanup_rec` | 本轮 cleanup 删除的 record 数 |
| `cleanup_counted` | 本轮 cleanup 删除的 counted track 缓冲数 |
| `released_crop` | 本轮释放的 best crop 数量 |

---

## 12. 健康运行状态参考

理想情况下：

```text
avg FPS > source FPS
writer_q 不长期接近队列上限
writer_drop 不持续快速增长
track_records 不长期单调增长
cleanup_rec 偶尔出现是正常的
程序结束后 CSV、视频、best crop 正常生成
```

如果出现以下情况，需要进一步检查：

```text
track_records 持续上涨
writer_q 长期接近队列上限
writer_drop 快速增长
avg FPS 低于视频原始 FPS
best_crops 明显缺失
confirmed_microbes.csv 行数异常
程序退出时卡在 writer.release()
```
