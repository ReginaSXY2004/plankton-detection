import threading
import queue


class AsyncVideoWriter:
    """
    异步视频写入器。

    主线程只负责 enqueue(frame)，
    后台线程负责真正的视频编码与写盘。

    目标：
    - 避免 writer.write() 阻塞主循环
    - 保证 realtime tracking/counting 优先
    - queue 满时允许丢弃 debug frame
    """

    def __init__(
        self,
        writer,
        max_queue_size=16,
    ):
        self.writer = writer

        self.queue = queue.Queue(maxsize=max_queue_size)

        self.running = True

        self.dropped_frames = 0
        self.written_frames = 0

        self.thread = threading.Thread(
            target=self._worker,
            daemon=True
        )

        self.thread.start()

    def _worker(self):
        while self.running or not self.queue.empty():

            try:
                frame = self.queue.get(timeout=0.1)

            except queue.Empty:
                continue

            self.writer.write(frame)

            self.written_frames += 1

            self.queue.task_done()

    def write(self, frame):
        """
        非阻塞 enqueue。

        queue 满时直接丢帧，
        保证 realtime 主循环不卡住。
        """

        if not self.running:
            return

        try:
            self.queue.put_nowait(frame.copy())

        except queue.Full:
            self.dropped_frames += 1

    def reopen(self, out_video_path):
        """
        segment 切换。

        先等待旧 queue 写完，
        再 reopen 新视频。
        """

        self.queue.join()

        self.writer.reopen(out_video_path)

    def stats(self):
        """
        返回异步 writer 当前状态，用于 runtime monitor。

        qsize:
            当前等待写入的视频帧数量

        dropped_frames:
            queue 满时被丢弃的帧数量

        written_frames:
            后台线程已经成功写入的帧数量
        """
        return {
            "qsize": self.queue.qsize(),
            "dropped_frames": self.dropped_frames,
            "written_frames": self.written_frames,
        }
    
    def release(self):
        """
        程序退出时：
        - 等待 queue 清空
        - 停止线程
        - release writer
        """

        self.queue.join()

        self.running = False

        self.thread.join()

        self.writer.release()