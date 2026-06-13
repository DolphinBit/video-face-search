"""视频段落截取模块 — 使用 ffmpeg 从视频中提取指定时间段"""
import subprocess
import json
from pathlib import Path


class SegmentExtractor:
    """使用 ffmpeg 截取视频段落"""

    def extract(
        self,
        video_path: str,
        segments: list[tuple[int, int, float]],
        output_dir: str,
    ) -> list[str]:
        """截取视频段落，保存为独立 mp4 文件。

        Args:
            video_path: 源视频路径
            segments: [(start_ms, end_ms, similarity), ...]
            output_dir: 输出目录

        Returns:
            生成的 mp4 文件路径列表
        """
        out = Path(output_dir)
        seg_dir = out / "segments"
        seg_dir.mkdir(parents=True, exist_ok=True)

        result_paths = []
        result_meta = []

        for i, (start_ms, end_ms, sim) in enumerate(segments, start=1):
            filename = f"segment_{i:03d}.mp4"
            output_path = seg_dir / filename

            start_s = start_ms / 1000.0
            duration_s = (end_ms - start_ms) / 1000.0

            cmd = [
                "ffmpeg", "-y",
                "-ss", str(start_s),
                "-i", str(video_path),
                "-t", str(duration_s),
                "-c", "copy",
                "-avoid_negative_ts", "make_zero",
                str(output_path),
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            result_paths.append(str(output_path))
            result_meta.append({
                "index": i,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "duration_s": round(duration_s, 3),
                "avg_similarity": sim,
                "file": filename,
            })

        # 写入 result.json
        meta_path = out / "result.json"
        meta_path.write_text(
            json.dumps(result_meta, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        return result_paths
