import os
import shutil
import subprocess
from typing import Optional

class FFmpegPipeWriter:
    """
    Stream raw BGR frames to ffmpeg via stdin.
    Tries HB_FFMPEG -> PATH -> imageio-ffmpeg. Falls back to CPU if NVENC missing.
    """
    def __init__(
        self,
        out_path,
        fps: int,
        size: tuple[int, int],  # (W, H)
        *,
        codec: str = "libx264",   # "libx264", "libx265", "h264_nvenc", "hevc_nvenc"
        crf: Optional[int] = 18,
        preset: str = "veryfast", # x264/x265: ultrafast..placebo; NVENC: p1..p7
        pix_fmt: str = "yuv420p",
        nvenc_cq: int = 19,
        nvenc_bitrate: str = "0",
        gpu: int = 0,
        faststart: bool = True,
        loglevel: str = "error",
    ):
        self.out_path = str(out_path)
        W, H = size

        ffmpeg = (
            os.environ.get("HB_FFMPEG")
            or shutil.which("ffmpeg")
            or self._imageio_ffmpeg_exe()
        )
        if not ffmpeg:
            raise RuntimeError(
                "ffmpeg not found. Set HB_FFMPEG=/path/to/ffmpeg or install ffmpeg, "
                "or run with HB_RENDER_WRITER=opencv."
            )

        # If user requested NVENC but encoder is missing, fall back to libx264.
        if codec in {"h264_nvenc", "hevc_nvenc"} and not self._has_encoder(ffmpeg, codec):
            print(f"[FFmpegPipeWriter] Encoder {codec} not available in {ffmpeg}. Falling back to libx264.")
            codec = "libx264"

        cmd = [
            ffmpeg, "-loglevel", loglevel, "-y",
            "-f", "rawvideo",
            "-pixel_format", "bgr24",
            "-video_size", f"{W}x{H}",
            "-framerate", str(fps),
            "-i", "-",
            "-an",
        ]

        if codec in {"h264_nvenc", "hevc_nvenc"}:
            cmd += [
                "-c:v", codec,
                "-preset", "p5" if preset == "veryfast" else preset,
                "-rc", "vbr",
                "-cq", str(nvenc_cq),
                "-b:v", nvenc_bitrate,
                "-gpu", str(gpu),
                "-pix_fmt", pix_fmt,
            ]
        else:
            cmd += ["-c:v", codec, "-preset", preset, "-pix_fmt", pix_fmt]
            if crf is not None and codec.startswith("libx26"):
                cmd += ["-crf", str(crf)]

        if faststart and self.out_path.lower().endswith((".mp4", ".m4v", ".mov")):
            cmd += ["-movflags", "+faststart"]

        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    @staticmethod
    def _imageio_ffmpeg_exe():
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None

    @staticmethod
    def _has_encoder(ffmpeg_path: str, encoder: str) -> bool:
        try:
            out = subprocess.run(
                [ffmpeg_path, "-hide_banner", "-encoders"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False
            ).stdout
            return encoder in out
        except Exception:
            return False

    def write(self, frame_bgr):
        if self.proc.stdin is None:
            raise RuntimeError("FFmpeg stdin closed")
        self.proc.stdin.write(frame_bgr.tobytes())

    def close(self):
        if self.proc.stdin:
            try:
                self.proc.stdin.flush()
                self.proc.stdin.close()
            except Exception:
                pass
        self.proc.wait()
