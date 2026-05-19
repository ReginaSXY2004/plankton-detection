import subprocess
import cv2


class VideoWriterWrapper:
    """
    统一封装不同视频写入后端。

    支持：
    1. opencv_mp4v：兼容性高，但通常走 CPU 编码
    2. ffmpeg_nvenc：Windows/NVIDIA PC 上使用 FFmpeg + NVENC
    3. jetson_gstreamer：Jetson GStreamer 基础硬件编码 pipeline
    4. jetson_gstreamer_bgrx：Jetson BGRx pipeline，历史优化版本
    5. jetson_gstreamer_bgr_queue：当前 Jetson 推荐 pipeline
       - 输入 OpenCV BGR frame
       - 由 GStreamer 内部完成颜色转换
       - 使用 nvv4l2h264enc 硬件编码
    6. none：不保存视频

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


    def _bitrate_to_int(self):
        """
        将 '8M' / '4000k' / 8000000 转成 GStreamer 需要的整数 bps。
        """
        if isinstance(self.bitrate, int):
            return self.bitrate

        s = str(self.bitrate).strip().lower()

        if s.endswith("m"):
            return int(float(s[:-1]) * 1_000_000)

        if s.endswith("k"):
            return int(float(s[:-1]) * 1_000)

        return int(float(s))

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



        elif self.backend == "jetson_gstreamer":
            bitrate = self._bitrate_to_int()

            gst_cmd = [
                "gst-launch-1.0",
                "-e",

                "fdsrc",
                "fd=0",

                "!",
                "rawvideoparse",
                f"width={self.width}",
                f"height={self.height}",
                f"framerate={int(round(self.fps))}/1",
                "format=bgr",

                "!",
                "videoconvert",

                "!",
                "video/x-raw,format=I420",

                "!",
                "nvvidconv",

                "!",
                "video/x-raw(memory:NVMM),format=NV12",

                "!",
                "nvv4l2h264enc",
                f"bitrate={bitrate}",
                "insert-sps-pps=true",
                "maxperf-enable=true",
                "preset-level=1",
                "iframeinterval=30",

                "!",
                "h264parse",

                "!",
                "qtmux",

                "!",
                "filesink",
                f"location={str(out_video_path)}",
            ]
            self.writer = subprocess.Popen(
                gst_cmd,
                stdin=subprocess.PIPE,
                stderr=None,
            )



        elif self.backend == "jetson_gstreamer_bgrx":
            bitrate = self._bitrate_to_int()

            gst_cmd = [
                "gst-launch-1.0",
                "-e",

                "fdsrc",
                "fd=0",

                "!",
                "rawvideoparse",
                f"width={self.width}",
                f"height={self.height}",
                f"framerate={int(round(self.fps))}/1",
                "format=bgrx",

                "!",
                "queue",
                "max-size-buffers=4",
                "leaky=downstream",

                "!",
                "nvvidconv",

                "!",
                "video/x-raw(memory:NVMM),format=NV12",

                "!",
                "nvv4l2h264enc",
                f"bitrate={bitrate}",
                "control-rate=1",
                "insert-sps-pps=true",
                "maxperf-enable=true",
                "preset-level=1",
                "iframeinterval=60",

                "!",
                "h264parse",

                "!",
                "qtmux",

                "!",
                "filesink",
                f"location={str(out_video_path)}",
            ]

        elif self.backend == "jetson_gstreamer_bgr_queue":
            bitrate = self._bitrate_to_int()

            gst_cmd = [
                "gst-launch-1.0",
                "-e",

                "fdsrc",
                "fd=0",

                "!",
                "rawvideoparse",
                f"width={self.width}",
                f"height={self.height}",
                f"framerate={int(round(self.fps))}/1",
                "format=bgr",

                "!",
                "queue",
                "max-size-buffers=4",
                "leaky=downstream",

                "!",
                "videoconvert",

                "!",
                "video/x-raw,format=I420",

                "!",
                "nvvidconv",

                "!",
                "video/x-raw(memory:NVMM),format=NV12",

                "!",
                "nvv4l2h264enc",
                f"bitrate={bitrate}",
                "control-rate=1",
                "insert-sps-pps=true",
                "maxperf-enable=true",
                "preset-level=1",
                "iframeinterval=60",

                "!",
                "h264parse",

                "!",
                "qtmux",

                "!",
                "filesink",
                f"location={str(out_video_path)}",
            ]

            self.writer = subprocess.Popen(
                gst_cmd,
                stdin=subprocess.PIPE,
                stderr=None,
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

        elif self.backend == "jetson_gstreamer":
            self.writer.stdin.write(frame.tobytes())

        elif self.backend == "jetson_gstreamer_bgrx":
            frame_bgrx = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            self.writer.stdin.write(memoryview(frame_bgrx))

        elif self.backend == "jetson_gstreamer_bgr_queue":
            self.writer.stdin.write(memoryview(frame))

    def release(self):
        if self.writer is None:
            return

        if self.backend == "opencv_mp4v":
            self.writer.release()

        elif self.backend in {
            "ffmpeg_nvenc",
            "jetson_gstreamer",
            "jetson_gstreamer_bgrx",
            "jetson_gstreamer_bgr_queue",
        }:
            self.writer.stdin.close()
            self.writer.wait()

        self.writer = None