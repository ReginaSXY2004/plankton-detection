# Plankton Detection Pipeline

浮游生物 Detection / Tracking / Counting 系统。

当前项目主要用于：

- 微生物实时检测（YOLOv8）
- 多目标 tracking
- realtime counting
- best crop 保存
- CSV 输出
- Jetson 边缘端部署

---

# 项目结构

```text
plankton-detection/
├── scripts/
├── src/
├── docs/
├── data/
├── runs/
└── README.md
```

---

# 当前部署环境

当前主要部署平台：

```text
Jetson AGX Orin 64GB
```

当前主要技术栈：

```text
YOLOv8
TensorRT FP16
OpenCV
GStreamer
Jetson NVENC
```

---

# 主入口

主运行入口：

```bash
PYTHONPATH=. python3 scripts/main.py
```

---

# 主要文档

推荐阅读顺序：

1. pipeline_overview.pdf
2. deployment_notes.pdf
3. output_structure.pdf
4. performance_notes.pdf
5. 数据标注与模型训练.pdf

---

# 当前系统特点

当前系统属于：

```text
Realtime AI Video Pipeline
```

核心设计原则：

```text
Realtime counting
>
Debug video 完整性
```

因此系统支持：

- async writer
- queue drop
- long-running cleanup
- segment video output

---

# 当前功能

支持：

- YOLO detection
- detection postprocess
- realtime tracker
- counting
- best crop
- CSV output
- runtime monitor
- Jetson hardware encoding

---

# 当前已知限制

当前仍存在：

- overlap organism tracking 困难
- intermittent detection
- 极密集目标场景性能下降
- rare class 数据不足

---

# Future Work

未来方向：

- 摄像头实时输入
- TensorRT INT8
- 更稳定 tracking
- 更强 writer pipeline
- 长时间无人值守运行
