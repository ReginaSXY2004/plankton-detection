import subprocess
import cv2


class VideoWriterWrapper:
    """
    统一封装不同视频写入后端。

    支持：
    1. opencv_mp4v：兼容性高，但通常走 CPU 编码
    2. ffmpeg_nvenc：Windows/NVIDIA 上使用 FFmpeg + NVENC 硬件编码
    3. none：不保存视频

    这个类也支持 reopen()，用于长时间运行时切分视频 segment。
    """

    def __init__(
        self,
        out_video_path,
        width: int,
        height: int,
        fps: float,
        backend: str,
        bitrate: str = "8M",
    ):
        self.width = width
        self.height = height
        self.fps = fps
        self.backend = backend
        self.bitrate = bitrate
        self.writer = None

        self.open(out_video_path)

    def open(self, out_video_path):
        self.out_video_path = out_video_path
        self.writer = None

        if self.backend == "none":
            return

        if self.backend == "opencv_mp4v":
            self.writer = cv2.VideoWriter(
                str(out_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                self.fps,
                (self.width, self.height)
            )

            if not self.writer.isOpened():
                raise RuntimeError("OpenCV VideoWriter 打不开")

        elif self.backend == "ffmpeg_nvenc":
            # 使用 FFmpeg pipe 将 BGR 原始帧传给 NVIDIA NVENC 硬件编码器。
            # 注意：这不是异步写入；如果 FFmpeg/NVENC 跟不上，stdin.write 仍然会阻塞。
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{self.width}x{self.height}",
                "-r", str(self.fps),
                "-i", "-",
                "-an",
                "-c:v", "h264_nvenc",
                "-b:v", self.bitrate,
                "-preset", "p1",
                "-tune", "ll",
                "-pix_fmt", "yuv420p",
                str(out_video_path),
            ]

            self.writer = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

        else:
            raise ValueError(f"Unknown video writer backend: {self.backend}")

    def reopen(self, out_video_path):
        """
        关闭当前视频文件，并打开一个新 segment。

        用于长时间部署：
        - 避免单个 mp4 文件过大
        - 降低异常退出导致整段视频损坏的风险
        - 方便后续按时间段回看
        """
        self.release()
        self.open(out_video_path)

    def write(self, frame):
        if self.writer is None:
            return

        if self.backend == "opencv_mp4v":
            self.writer.write(frame)

        elif self.backend == "ffmpeg_nvenc":
            self.writer.stdin.write(frame.tobytes())

    def release(self):
        if self.writer is None:
            return

        if self.backend == "opencv_mp4v":
            self.writer.release()

        elif self.backend == "ffmpeg_nvenc":
            self.writer.stdin.close()
            self.writer.wait()

        self.writer = None