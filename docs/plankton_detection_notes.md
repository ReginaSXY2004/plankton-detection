# 🧪 plankton_detection_notes

## 🔄 整体流程

```text
逐帧读视频
→ YOLO 检测
→ 去重 / 杂质过滤
→ tracker 关联 ID
→ 筛选稳定轨迹
→ 确认（计数）
→ 选最佳 crop
→ 输出结果（CSV / 图片）
```

---

## 1️⃣ 检测（Detection）

使用 YOLO 对每一帧做检测：

输入：

* 一帧图像

输出：

* bbox（位置）
* conf（置信度）

👉 本质：
在当前帧中找出“像微生物的目标”

---

## 2️⃣ 过滤（Cleaning）

两步：

### （1）去重（dedup）

* 同一个目标可能被检测出多个框
  → 保留一个

### （2）杂质过滤（blob filter）

* 过滤：

  * 太圆
  * 太光滑
  * 像光斑的目标

👉 本质：
减少重复检测和明显误检

---

## 3️⃣ 跟踪（Tracking）

使用 `MicrobeTracker`：

作用：

* 把当前帧的检测和上一帧对应起来

依据：

* 距离
* IoU
* 尺寸
* 运动方向

结果：

* 能匹配 → 同一个 `track_id`
* 匹配不上 → 新建 `track_id`

👉 本质：
判断是不是同一只微生物

---

## 4️⃣ 可见轨迹筛选（visible_tracks）

只保留比较稳定的轨迹：

```python
hits >= min_hits_to_show
and
missed <= 1
```

👉 去掉：

* 只出现1~2帧的短命ID

👉 本质：
第一层过滤（稳定性）

---

## 5️⃣ 确认（confirmed）

当：

```python
hits >= CONFIRM_MIN_HITS
```

认为这是一个真实个体：

执行：

* 分配 `display_id`
* 计数 +1（只加一次）

👉 本质：
第二层过滤（真正参与计数的对象）

---

## 6️⃣ 最佳截图（best crop）

不是每帧保存，而是：

每帧做：

* 按 bbox 裁剪
* 计算清晰度（Laplacian）
* 结合 conf / 尺寸

如果更好：
→ 更新 best_crop

当 track 结束时：
→ 只保存一张 best_crop

👉 本质：
为每个微生物选一张最清晰的代表图

---

## 7️⃣ 计数（Count）

计数方式：

* 每个 track 只计一次
* 在变成 confirmed 时 +1

不使用：

* 检测框数量
* track_id 总数

👉 本质：
统计“独立个体数量”

---

## 8️⃣ 输出（Output）

默认只保留：

* confirmed CSV（每个微生物一行）
* best crop（每个微生物一张图）

关闭：

* 每帧 debug CSV
* debug 视频

👉 本质：
减少 IO，提高实时性

---
