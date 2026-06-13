import numpy as np
import pytest
from src.scanner import VideoScanner, ScanResult


class TestVideoScanner:
    def test_scan_returns_list(self, test_video_path):
        """扫描视频应返回 ScanResult 列表"""
        scanner = VideoScanner(frame_interval=5)
        results = scanner.scan(test_video_path)
        assert isinstance(results, list)
        for r in results:
            assert isinstance(r, ScanResult)
            assert isinstance(r.timestamp_ms, int)
            assert isinstance(r.embedding, np.ndarray)
            assert r.embedding.shape == (512,)
            assert r.embedding.dtype == np.float32

    def test_scan_with_progress_callback(self, test_video_path):
        """进度回调应被调用"""
        progress_values = []
        scanner = VideoScanner(frame_interval=5)

        def on_progress(pct, msg):
            progress_values.append(pct)

        scanner.scan(test_video_path, progress_callback=on_progress)
        assert len(progress_values) > 0
        assert progress_values[-1] == 100

    def test_scan_nonexistent_file_raises(self):
        """不存在的视频文件应抛出异常"""
        scanner = VideoScanner()
        with pytest.raises(FileNotFoundError):
            scanner.scan("/nonexistent/video.mp4")

    def test_frame_interval_respected(self, test_video_path):
        """帧采样间隔应正确生效"""
        # 10fps 视频，3秒 = 30帧，间隔5 → 约6帧被采样
        scanner = VideoScanner(frame_interval=5)
        results = scanner.scan(test_video_path)
        assert len(results) <= 7
