"""视频扫描模块 — 抽帧 + 人脸检测 + 嵌入提取"""
from dataclasses import dataclass
import numpy as np
import cv2
from src.embedder import FaceEmbedder


@dataclass
class ScanResult:
    """单帧扫描结果"""
    timestamp_ms: int       # 帧在视频中的时间位置（毫秒）
    embedding: np.ndarray   # 512维人脸嵌入向量


class VideoScanner:
    """逐帧采样视频，检测人脸并提取嵌入向量"""

    def __init__(self, frame_interval: int = 5):
        self.frame_interval = frame_interval
        self.embedder = FaceEmbedder()

    def scan(
        self,
        video_path: str,
        progress_callback: callable = None,
        cancel_flag: callable = None,
    ) -> list[ScanResult]:
        """扫描视频，返回所有检测到的人脸的嵌入向量。"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        results = []
        frame_idx = 0

        while True:
            if cancel_flag and cancel_flag():
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % self.frame_interval == 0:
                faces = self.embedder.extract_from_array(frame)
                timestamp_ms = int(frame_idx / fps * 1000)

                for face in faces:
                    results.append(ScanResult(
                        timestamp_ms=timestamp_ms,
                        embedding=face["embedding"],
                    ))

                if progress_callback and total_frames > 0:
                    pct = int(frame_idx / total_frames * 100)
                    progress_callback(pct, f"正在扫描... {pct}%")

            frame_idx += 1

        cap.release()

        if progress_callback:
            progress_callback(100, "扫描完成")

        return results
