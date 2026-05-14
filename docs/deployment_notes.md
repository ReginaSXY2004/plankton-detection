# 边缘端部署说明（Jetson / Linux）

本文档用于实验室内部部署与交接，说明如何将当前浮游生物检测与追踪项目部署到 Jetson / Linux 边缘端。

---

## 1. 当前部署目标

当前阶段的目标是先验证：

- 代码能否在 Jetson / Linux 上正常运行
- 离线视频能否完成推理、追踪、计数和输出
- 长时间运行时是否稳定
- `track_records` 是否会长期增长
- 视频写入队列是否积压
- CSV、best crop、debug video 是否正常生成

推荐顺序：

```text
Windows 本地开发
→ GitHub 同步代码
→ Jetson / Linux 拉取代码
→ 放置模型和测试视频
→ 跑离线视频
→ 跑长时间压力测试
→ 后续再接入实时摄像头
```

---

## 2. 推荐目录结构

Jetson / Linux 端建议保持与 Windows 本地项目一致的目录结构：

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

当前代码中大量路径基于 `PROJECT_ROOT` 拼接，因此只要项目内部目录结构一致，不需要保持 Windows 的绝对路径。

---

## 3. Git 管理说明

GitHub 仓库建议保存：

- Python 源代码
- 配置文件
- README
- docs 文档
- `.gitignore`
- requirements / 环境说明

GitHub 仓库不建议保存：

- 原始视频
- 输出视频
- 训练数据
- 模型权重
- TensorRT engine
- `runs/track_analysis/` 运行结果

这些大文件应手动放到 Jetson 本地对应目录。

---

## 4. 模型与测试视频放置位置

### 模型文件

推荐放置路径：

```text
runs/yolov8n_1x_multiclass_v1/weights/
```

示例：

```text
runs/yolov8n_1x_multiclass_v1/weights/best.pt
runs/yolov8n_1x_multiclass_v1/weights/best.engine
```

注意：Windows 上生成的 `.engine` 通常不能直接复制到 Jetson 使用。

### 测试视频

推荐放置路径：

```text
data/video1x/
```

示例：

```text
data/video1x/sample.avi
data/video1x/sample16.avi
data/video1x/sample17.avi
```

---

## 5. TensorRT engine 注意事项

TensorRT `.engine` 文件通常与以下因素强绑定：

- GPU 型号
- CUDA 版本
- TensorRT 版本
- 操作系统
- CPU 架构

因此：

```text
Windows RTX 电脑导出的 best.engine
通常不能直接拿到 Jetson 上运行
```

推荐流程：

```text
Windows / 本地电脑：
保留 best.pt

Jetson：
使用 best.pt 在 Jetson 上重新 export TensorRT engine
```

---

## 6. 当前输入方式

当前项目使用离线视频输入：

```python
cv2.VideoCapture(VIDEO_PATH)
```

也就是：

```text
提前录好的视频
→ 逐帧读取
→ YOLO 检测
→ tracker 追踪
→ CSV / video / best crop 输出
```

当前尚未正式接入：

- USB 实时摄像头
- CSI 摄像头
- RTSP 网络流

后续接摄像头时，核心主循环仍可复用，但输入源需要替换。

---

## 7. 视频写入 backend

当前 `VIDEO_WRITER_BACKEND` 已支持：

```text
opencv_mp4v
ffmpeg_nvenc
none
```

说明：

- `opencv_mp4v`：兼容性较高，但通常是 CPU 编码
- `ffmpeg_nvenc`：适合 Windows / NVIDIA PC，通过 FFmpeg + NVENC 硬件编码
- `none`：不保存视频，只跑检测、追踪和 CSV

注意：

```text
jetson_gstreamer backend 目前只是预留方向，尚未完成正式实现。
```

Jetson 上后续更可能使用：

```text
GStreamer
nvv4l2h264enc
```

---

## 8. 推荐部署工作流

### Windows 本地开发

```text
修改代码
→ 本地测试
→ git add
→ git commit
→ git push
```

### Jetson / Linux 部署

```text
Jetson 开机
→ SSH 连接 Jetson
→ 进入项目目录
→ git pull
→ 激活 Jetson 本地 Python 环境
→ 检查模型和视频是否放到正确目录
→ 运行 main.py
```

示例流程：

```bash
cd ~/plankton-detection
git pull
source venv/bin/activate
python scripts/main.py
```

---

## 9. 环境说明

Windows 本地的 `venv312` 不建议、也通常不能直接迁移到 Jetson。

原因：

```text
Windows: x86_64 + Windows + RTX GPU
Jetson: ARM64 + Linux + Jetson CUDA/TensorRT
```

Jetson 上需要重新创建环境并安装适配版本的依赖。

---

## 10. 长时间运行重点观察指标

运行时终端会打印 runtime monitor。

重点关注：

```text
avg FPS
window FPS
active_tracks
visible_tracks
track_records
writer_q
writer_drop
writer_written
cleanup_rec
cleanup_counted
released_crop
```

理想状态：

- `avg FPS` 高于输入视频 FPS
- `track_records` 不长期单调增长
- `writer_q` 不长期接近队列上限
- `writer_drop` 不持续快速增长
- `cleanup_rec` 偶尔出现是正常现象
- `released_crop` 偶尔出现表示 best crop 内存被释放
- 程序结束后 CSV、视频、best crop 都能正常保存

---

## 11. 当前推荐测试顺序

建议部署前后按以下顺序测试：

```text
1. Windows 本地短视频测试
2. Windows 本地 30 分钟循环压力测试
3. Jetson 离线短视频测试
4. Jetson 30 分钟压力测试
5. Jetson 1 小时以上压力测试
6. 后续再接入实时摄像头
```

---

## 12. 当前项目重点

当前工程重点：

- 检测与追踪稳定性
- 重复计数抑制
- best crop 保存质量
- 长时间运行内存释放
- 异步视频写入
- 视频 segment 切分
- 边缘端部署稳定性