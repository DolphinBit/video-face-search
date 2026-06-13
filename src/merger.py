"""视频合并模块 — 使用 ffmpeg concat 拼接多个视频文件"""
import subprocess
import tempfile
from pathlib import Path


class VideoMerger:
    """使用 ffmpeg concat demuxer 合并视频"""

    def merge(self, segment_paths: list[str], output_path: str) -> str | None:
        """将多个视频段落合并为一个视频文件。

        Args:
            segment_paths: 要合并的视频文件路径列表
            output_path: 输出文件路径

        Returns:
            合并后的文件路径，如果输入列表为空则返回 None
        """
        if not segment_paths:
            return None

        # 写入 concat 文件列表
        concat_file = tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        )
        try:
            for p in segment_paths:
                abs_path = str(Path(p).resolve()).replace("\\", "/")
                concat_file.write(f"file '{abs_path}'\n")
            concat_file.close()

            cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", concat_file.name,
                "-c", "copy",
                str(output_path),
            ]
            subprocess.run(cmd, capture_output=True, check=True)

        finally:
            Path(concat_file.name).unlink(missing_ok=True)

        return output_path
