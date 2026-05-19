# 性能分析与优化说明

本文档用于说明当前系统的 FPS 构成、性能瓶颈、writer pipeline、异步写入设计以及长期运行相关问题。

---

# 1. 当前系统特点

当前系统属于：

```text
Realtime AI Video Pipeline
```

主要包含：

- YOLO detection
- detection postprocess
- tracking
- counting
- visualization
- video writing

系统目标：

```text
尽量接近 realtime
```

---

# 2. FPS 主要组成

单帧时间通常由：

```text
YOLO
postprocess
tracking
draw
video write
```

组成。

典型结构：

```text
cap.read()
→ YOLO
→ deduplicate
→ blob filter
→ tracker.update()
→ draw
→ writer.write()
```

---

# 3. 当前主要瓶颈

当前系统最大的性能瓶颈通常是：

# 视频编码与 IO

而不是：

```text
tracking
postprocess
```

---

# 4. 当前 Jetson 实测性能

当前 Jetson AGX Orin 实测：

```text
YOLO TensorRT FP16:
约 25~30ms

Tracking/Postprocess:
通常 < 3ms

Jetson GStreamer Writer:
约可稳定支持 30~34 FPS realtime

test_writer_speed.py:
writer 极限吞吐约 80+ FPS
```

当前系统已经能够：

```text
接近 realtime 视频处理
```

---

# 5. YOLO 推理耗时

YOLO detection 通常占：

```text
20~30ms
```

主要受：

- model size
- imgsz
- TensorRT
- 输入分辨率

影响。

当前系统：

```python
imgsz = 800
```

---

# 6. 视频写入瓶颈

视频写入通常是：

```text
最容易导致 FPS 下降的部分
```

原因包括：

- 编码
- 磁盘 IO
- pipe 阻塞
- CPU/GPU backend 差异

典型表现：

```text
不保存视频：
FPS 很高

保存视频：
FPS 明显下降
```

这是正常现象。

---

# 7. AsyncVideoWriter 设计

当前系统使用：

```text
AsyncVideoWriter
```

核心思想：

```text
主线程：
enqueue(frame)

后台线程：
真正执行编码与写盘
```

目标：

```text
tracking/counting
不被 writer 阻塞
```

---

# 8. 为什么允许 drop frame

当前系统：

```text
允许 queue 满时丢弃 debug frame
```

原因：

```text
Realtime counting
优先级
高于 debug video 完整性
```

否则：

```text
writer 阻塞
→ 主循环卡住
→ realtime tracking 失效
```

---

# 9. Writer Queue

当前 writer 采用：

```python
queue.Queue(maxsize=64)
```

运行时重点观察：

```text
qsize
dropped_frames
written_frames
```

---

## qsize

表示：

```text
当前等待写入的视频帧数量
```

长期增长说明：

```text
writer 跟不上
```

---

## dropped_frames

表示：

```text
queue 满时被丢弃的帧
```

少量 drop：

```text
可接受
```

持续增长：

```text
writer pipeline 已成为瓶颈
```

---

## written_frames

表示：

```text
后台线程真正成功写入的视频帧数
```

---

# 10. SAVE_VIDEO_EVERY_N_FRAMES

当前系统支持：

```text
1 = 每帧写
2 = 每2帧写1帧
3 = 每3帧写1帧
```

作用：

```text
降低 writer 压力
```

代价：

```text
视频观感变卡
```

因此：

```text
更适合作为性能测试开关
```

而不是最终部署方案。

---

# 11. Writer Backend 的优化过程

## 11.1：OpenCV mp4v

```text
Python
→ OpenCV VideoWriter(mp4v)
→ CPU encode
```

问题：

- FPS 下降明显
- CPU 占用高
- 长时间 realtime 不稳定

---

## 11.2： Jetson GStreamer + BGRx

```text
Python BGR
→ cv2.cvtColor(BGR→BGRA)
→ GStreamer bgrx
→ nvvidconv
→ nvv4l2h264enc
```

优点：

- 使用 NVENC
- writer 明显变快

问题：

- Python 仍需每帧颜色转换
- CPU 额外 copy/convert 开销


---

## 11.3：jetson_gstreamer_bgr_queue（当前版本）

```text
Python BGR
→ queue
→ videoconvert
→ nvvidconv
→ nvv4l2h264enc
```

核心改进：

```text
将颜色转换放入 GStreamer pipeline 内部
避免 Python 侧 cv2.cvtColor(BGR→BGRA)
```

结果：

```text
writer_drop 显著下降
writer_q 更稳定
33FPS 原视频可实现 realtime 全帧保存
```

*当前版本并不代表理论最优，是“稳定性 / 可维护性 / realtime 性能”综合权衡后的版本。或许仍有优化空间

---

# 12. test_writer_speed.py

当前项目提供：

```text
test_writer_speed.py
```

用途：

```text
单独测试 writer pipeline 极限吞吐
```

特点：

- 不运行 YOLO
- 不运行 tracking
- 不画框
- 纯测试 writer

核心用途：

```text
判断瓶颈是否来自 writer
```

注意：

```text
writer_test.py 的 FPS
不代表整个系统最终 FPS
```

因为：

```text
main.py 还包含：
YOLO
tracking
draw
CSV
crop update
```

---

# 13. 为什么 tracking 不是主要瓶颈

当前 tracker 属于：

```text
轻量 tracker
```

主要计算：

- distance
- IoU
- size ratio
- velocity extrapolation

整体耗时通常远小于：

```text
YOLO
video encoding
```

---

# 14. 大量目标导致性能下降的原因

当画面中微生物数量大量增加时：

可能导致：

- draw 开销增加
- tracking 匹配量增加
- crop/CSV/update 次数增加
- writer 编码复杂度增加

其中：

```text
YOLO latency 通常变化不大
```


而 
```text
tracking / draw / crop update
```
会随着目标数量增加而明显增长。

---

# 15. 长时间运行风险

## 1. track_records 增长

如果不 cleanup：

```text
内存会持续增长
```

当前系统：

```text
finalize 后主动释放
```

---

## 2. best crop 内存占用

如果保存所有 crop：

```text
RAM 会持续增加
```

因此：

```text
每个 track 只保留当前 best crop
```

---

## 3. 单个视频文件过大

当前：

```text
segment 视频切分
```

避免：

- 超大 mp4
- 单文件损坏风险

---

# 16. 当前优化方向

## 1. TensorRT

目标：

```text
降低 YOLO latency
```

---

## 2. 更高效 writer pipeline

例如：

- Jetson GStreamer
- 更稳定 NVENC
- 更少 pipe 阻塞

---

## 3. 更高效 tracking

未来可能：

- 更稳定 reconnect
- 更复杂 motion model

但当前：

```text
tracking 不是主要性能瓶颈
```

---

# 17. 当前工程设计思想

当前系统核心设计原则：

```text
Realtime counting
>
Debug video 完整性
>
视觉观感
```

因此系统设计上：

- 允许 writer drop
- 使用 async writer
- 主动 cleanup
- segment 切分
- best crop 只保留一张

这些设计本质上：

```text
是为了长期稳定运行
```
