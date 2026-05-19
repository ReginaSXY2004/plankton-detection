# 边缘端部署说明（Jetson / Linux）

本文档用于实验室内部部署与后续维护交接，说明当前浮游生物检测与追踪系统在 Jetson / Linux 边缘端的部署方式、运行结构、长期运行设计与已知工程风险。

---

# 1. 项目用途

当前系统主要用于：

- 浮游生物视频检测
- 多目标 tracking 与计数
- best crop 保存
- CSV 统计输出
- 长时间稳定运行
- Jetson 边缘端部署

当前阶段以：

```text
离线视频推理
```

为主。

未来计划扩展：

```text
实时摄像头输入
自动采集
边缘端长期运行
机器人系统集成
```

---

# 2. 当前系统结构

当前主流程：

```text
Video Input
→ YOLO Detection
→ Detection Postprocess
→ Microbe Tracker
→ Counting
→ Best Crop Update
→ CSV / Video Output
→ Long-running Cleanup
```

主入口：

```text
scripts/main.py
```

---

# 3. 推荐目录结构

推荐保持如下目录结构：

```text
plankton-detection/
├── scripts/
├── src/
├── docs/
├── data/
│   └── video1x/
├── runs/
│   └── yolov8n_1x_multiclass_v1/
│       └── weights/
└── README.md
```

当前系统大量使用：

```python
PROJECT_ROOT / ...
```

进行相对路径拼接。

因此：

```text
不要随意修改项目内部目录结构。
```

---

# 4. Git 管理建议

建议 GitHub 仓库仅保存：

- Python 源代码
- docs 文档
- 配置文件
- requirements
- README

不建议保存：

- 原始视频
- 输出视频
- tracking 结果
- TensorRT engine
- 大型训练数据

推荐：

```text
代码走 GitHub
数据与模型本地保存
```

---

# 5. Python 与依赖环境

当前项目属于：

```text
GPU + TensorRT + OpenCV + GStreamer
```

类型工程。

环境兼容性非常重要。

不同版本之间可能存在：

- TensorRT engine 不兼容
- CUDA 不兼容
- OpenCV backend 差异
- GStreamer pipeline 差异

---

## 当前已验证环境（Verified Environment）

```text
Device:
NVIDIA Jetson AGX Orin 64GB

Python:
3.10

PyTorch:
2.3.0

Ultralytics:
8.4.50

OpenCV:
4.13.0

CUDA:
JetPack bundled CUDA

TensorRT:
JetPack bundled TensorRT

JetPack:
6.x

GStreamer:
Jetson system GStreamer
```

建议后续修改环境前：

- 先备份当前可运行版本
- 重新验证 TensorRT engine
- 重新测试 writer pipeline

---

# 6. 模型文件放置位置

推荐：

```text
runs/yolov8n_1x_multiclass_v1/weights/
```

示例：

```text
best.pt
best.engine
best_jetson_fp16.engine
```

---

# 7. TensorRT Engine 注意事项

TensorRT engine 文件通常与：

- GPU
- CUDA
- TensorRT
- CPU 架构
- 操作系统

强绑定。

因此：

```text
Windows RTX 导出的 .engine
通常不能直接在 Jetson 使用。
```

推荐流程：

```text
Windows:
保留 best.pt

Jetson:
重新 export TensorRT engine
```

---

# 8. 当前输入方式

当前系统主要输入方式：

```python
cv2.VideoCapture(VIDEO_PATH)
```

即：

```text
离线视频
→ 逐帧读取
→ detection
→ tracking
```

当前尚未正式接入：

- USB Camera
- CSI Camera
- RTSP Stream

未来可替换输入源，但主循环逻辑基本可复用。

---

# 9. 视频写入 Backend

当前支持：

```text
opencv_mp4v
ffmpeg_nvenc
jetson_gstreamer
jetson_gstreamer_bgr_queue
none
```

---

## opencv_mp4v

优点：

- 兼容性高
- 简单稳定

缺点：

- 通常为 CPU 编码
- FPS 较低

---

## ffmpeg_nvenc

适用于：

```text
Windows + NVIDIA GPU
```

优点：

- 使用 NVENC 硬件编码
- 明显降低 CPU 压力

缺点：

- 依赖 FFmpeg
- Jetson 上通常不可直接复用

---

## jetson_gstreamer_bgr_queue（当前推荐 backend）

适用于：

```text
Jetson Linux
```

基于：

```text
Jetson GStreamer + NVIDIA Hardware Encoder
```

当前为：

```text
Jetson 实测最稳定 backend
```

核心 pipeline：

```text
BGR frame
→ queue
→ videoconvert
→ nvvidconv
→ nvv4l2h264enc
→ h264 hardware encode
→ mp4
```

该 backend 是当前阶段：

```text
长期 realtime tracking/counting
```

表现最稳定的方案。

---

## none

不保存视频。

适用于：

```text
纯性能测试
仅测试 detection/tracking
```

---

# 10. Async Writer 设计

系统当前采用：

```text
AsyncVideoWriter
```

主线程：

```text
enqueue(frame)
```

后台线程：

```text
真正执行视频编码与写盘
```

系统设计思想：

```text
Realtime counting
>
Debug video 完整性
```

因此：

```text
queue 满时允许 drop frame
```

这是主动设计行为。

当前系统本质属于：

```text
Realtime AI Pipeline
```

而不是：

```text
离线视频渲染系统
```

---

# 11. 长时间运行设计

当前系统已开始支持：

```text
长时间连续运行
```

主要风险包括：

- track_records 无限增长
- best_crop 长期占用内存
- writer queue 堆积
- 单个 mp4 文件过大
- 异常退出导致视频损坏

---

## cleanup 机制

系统会主动释放：

- finalized track
- old best crop
- inactive records

否则长期运行可能导致：

- RAM 持续增长
- Python dict 无限增大
- best crop 长期占用内存

---

## segment 视频切分

当前：

```python
VIDEO_SEGMENT_MINUTES = 30
```

长时间运行时：

```text
自动切分新视频
```

避免：

- 超大 mp4
- 单文件损坏风险
- 长时间写盘异常

---

# 12. Jetson 部署建议

推荐流程：

```text
Windows 本地开发
→ GitHub push
→ Jetson git pull
→ Jetson 本地生成 TensorRT engine
→ 长时间运行测试
→ 后续接入摄像头
```

建议优先完成：

1. 离线视频稳定运行
2. 长时间稳定运行
3. writer backend 稳定
4. cleanup 正常工作

之后再接：

```text
实时摄像头
机器人系统
```

---

# 13. 当前已知风险

## 1. overlapping organisms

两个微生物重叠时：

- 可能发生 ID switch
- reconnect 可能错误接回

---

## 2. intermittent detection

YOLO detection 可能不连续。

表现：

```text
短暂消失
再次出现
```

tracker 已通过：

```text
reconnect
missed
velocity extrapolation
```

进行缓解，但无法完全消除。

---

## 3. writer queue overflow

当：

```text
视频写入速度 < 推理速度
```

时：

```text
queue 会积压
```

当前允许：

```text
drop debug frame
```

以保证：

```text
Realtime counting
```

优先稳定运行。

---

# 14. Future Work

未来方向：

- 实时摄像头输入
- Jetson TensorRT FP16 / INT8
- 更稳定 tracking
- 多倍率模型自动切换
- 更高 FPS writer pipeline
- 长时间无人值守运行
- 边缘机器人集成
