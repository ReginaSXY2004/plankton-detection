# Plankton Detection

基于 YOLO 与自定义 Tracker 的浮游生物检测、追踪与计数系统。

---

## Goals

- 从显微视频中检测浮游生物
- 对目标进行实时追踪（tracking）
- 减少重复计数与 ID 断裂
- 自动保存最佳 crop 图像
- 输出统计 CSV 与可视化视频
- 支持不同倍率显微视频分析

---

## Pipeline

```text
Video Input
→ YOLO Detection
→ Detection Postprocess
→ MicrobeTracker
→ Stable Track Filtering
→ Confirmed Counting
→ Best Crop Selection
→ CSV / Video Output
```

---

## Features

### Detection

- YOLOv8 based microbe detection
- Multi-class plankton classification
- Detection deduplication
- Blob / reflection filtering

### Tracking

- Custom MicrobeTracker
- Lost track reconnect
- No-spawn zone
- Direction-aware matching
- Duplicate track suppression

### Counting

- Confirmed-track based counting
- Realtime counting
- Duplicate-count reduction

### Best Crop

- Sharpness-based crop scoring
- Automatic best-frame selection
- One representative image per microbe

### Output

- Realtime visualization
- Confirmed CSV export
- Best crop saving
- Debug video support

---

## Current Focus

Current development focuses on:

- Real water environment robustness
- Tracking stability
- Long-term realtime deployment
- Multi-magnification adaptation
- Edge-device deployment optimization
