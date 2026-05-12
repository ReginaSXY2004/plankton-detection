import subprocess
import cv2


class VideoWriterWrapper:
    def __init__(
        self,
        out_video_path,
        width: int,
        height: int,
        fps: float,
        backend: str,
        bitrate: str = "8M",
    ):
        self.backend = backend
        self.writer = None

        if backend == "none":
            return

        if backend == "opencv_mp4v":
            self.writer = cv2.VideoWriter(
                str(out_video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height)
            )

            if not self.writer.isOpened():
                raise RuntimeError("OpenCV VideoWriter 打不开")

        elif backend == "ffmpeg_nvenc":
            ffmpeg_cmd = [
                "ffmpeg",
                "-y",
                "-f", "rawvideo",
                "-vcodec", "rawvideo",
                "-pix_fmt", "bgr24",
                "-s", f"{width}x{height}",
                "-r", str(fps),
                "-i", "-",
                "-an",
                "-c:v", "h264_nvenc",
                "-b:v", bitrate,
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
            raise ValueError(f"Unknown video writer backend: {backend}")

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