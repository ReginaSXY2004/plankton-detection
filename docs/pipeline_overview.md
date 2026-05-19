# Pipeline Overview

## 1. 项目目标

本项目用于：

- 浮游生物实时检测（Detection）
- 浮游生物轨迹追踪（Tracking）
- 实时计数（Counting）
- 最佳截图保存（Best Crop）
- CSV 数据记录
- Jetson 边缘设备实时部署

当前系统重点：

```text
Realtime First
```

即：

```text
实时检测稳定性
>
Debug 视频完整性
```

---

# 2. 整体 Pipeline

```text
Video Input
↓
YOLO Detection
↓
Detection Postprocess
↓
Tracking
↓
Realtime Counting
↓
Best Crop Update
↓
Async Video Writer
↓
CSV / Crop / Video Output
```

---

# 3. 模块说明

## 3.1 YOLO Detection

负责：

- 微生物目标检测
- bbox 输出
- class 输出
- confidence 输出

当前部署：

```text
YOLOv8 + TensorRT FP16
```

运行平台：

```text
Jetson AGX Orin 64G
```

---

## 3.2 Detection Postprocess

负责：

- detection 去重
- bbox 清理
- 低质量 detection 过滤

主要用于减少：

- 重复框
- 噪声 detection
- tracking 抖动

---

## 3.3 Tracking

负责：

- detection 与历史 track 匹配
- track 生命周期维护
- ID 稳定
- reconnect

当前 tracking 为：

```text
轻量级 realtime tracker
```

设计目标：

```text
低延迟
>
复杂轨迹预测
```

---

## 3.4 Realtime Counting

confirmed track 首次出现时：

```text
count += 1
```

避免：

- 同一微生物重复计数
- detection 抖动导致重复计数

---

## 3.5 Best Crop Update

每个 confirmed track：

- 保存最佳截图
- 更新 sharpness / confidence / area score

当前 best crop 属于：

```text
heuristic scoring
```

主要用于：

- 后续人工检查
- 数据集积累
- 微生物分析

---

## 3.6 Async Video Writer

当前 writer：

```text
Python BGR
→ queue
→ videoconvert
→ nvvidconv
→ nvv4l2h264enc
→ H264 hardware encode
→ mp4
```

当前使用：

```text
Jetson NVENC hardware encoder
```

特点：

- realtime optimized
- async queue writer
- 可允许 debug frame drop
- 避免阻塞主 inference pipeline

---

## 3.7 Cleanup System

长期运行时：

- finalized track 会释放
- best crop 会写入磁盘
- CSV 会写入磁盘
- 内存中的历史记录会清理

目标：

```text
支持长期无人值守运行
```

---

# 4. 当前系统特点

## Realtime First

系统优先保证：

```text
Detection / Tracking realtime
```

而不是：

```text
Debug video 完整性
```

---

## Edge Deployment

当前系统主要面向：

```text
Jetson Edge AI Deployment
```

---

## Long-running Friendly

系统设计考虑：

- 长时间运行
- segment 输出
- 内存释放
- async writer
- queue overflow

---

# 5. 当前主要 Tradeoff

## Writer 可丢帧

当 writer queue 满时：

```text
允许丢弃 debug frame
```

原因：

```text
Realtime counting
>
Debug video 完整性
```

---

## Tracking 不保证完全正确

复杂情况：

- overlap organism
- detection intermittent
- motion blur

仍可能导致：

- ID swap
- track fragmentation

---

## Best Crop 属于经验规则

当前 best crop 评分：

- sharpness
- confidence
- bbox area

属于经验性人为打分逻辑。

后续可继续优化。

---

# 6. 当前性能（Jetson）

当前典型性能：

```text
YOLO TensorRT FP16
≈ 33 FPS realtime inference
```

当前 writer：

```text
支持 realtime 全帧保存
```

---

# 7. 当前 Limitations

当前仍存在：

- overlap organism tracking 困难
- 多倍率 domain shift
- rare class 数据不足
- 检测断断续续
- 小目标极端密集场景

---

# 8. 后续优化方向

可能的后续方向：

- true NVMM zero-copy pipeline
- TensorRT INT8
- 更强 tracking
- 多相机输入
- 湖泊真实数据重新训练
- 更稳定的小目标检测
