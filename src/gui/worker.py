"""后台扫描工作线程 — 避免阻塞 GUI"""
from PyQt6.QtCore import QThread, pyqtSignal
from src.embedder import FaceEmbedder
from src.scanner import VideoScanner
from src.matcher import Matcher


class ScanWorker(QThread):
    """后台执行：提取查询嵌入 → 扫描视频 → 匹配段落"""

    # 信号
    progress = pyqtSignal(int, str)      # percent, message
    finished = pyqtSignal(list)           # [(start_ms, end_ms, score), ...]
    error = pyqtSignal(str)               # error message

    def __init__(
        self,
        face_image_path: str,
        video_path: str,
        frame_interval: int = 5,
        threshold: float = 0.55,
        merge_mode: str = "auto",
        merge_gap: float = 2.0,
        fixed_seconds: float = 3.0,
        min_segment_duration: float = 1.0,
        parent=None,
    ):
        super().__init__(parent)
        self.face_image_path = face_image_path
        self.video_path = video_path
        self.frame_interval = frame_interval
        self.threshold = threshold
        self.merge_mode = merge_mode
        self.merge_gap = merge_gap
        self.fixed_seconds = fixed_seconds
        self.min_segment_duration = min_segment_duration
        self._cancel = False

    def cancel(self):
        """请求取消扫描"""
        self._cancel = True

    def run(self):
        try:
            # 步骤 A: 提取查询人脸嵌入
            self.progress.emit(0, "正在分析查询人脸...")
            embedder = FaceEmbedder()
            query_emb = embedder.extract_embedding(self.face_image_path)

            if query_emb is None:
                self.error.emit("查询图片中未检测到人脸")
                return

            # 步骤 B: 扫描视频
            self.progress.emit(5, "正在扫描视频中的人脸...")
            scanner = VideoScanner(frame_interval=self.frame_interval)

            def scan_progress(pct, msg):
                # 映射到 5%-85% 区间
                mapped = int(5 + pct * 0.8)
                self.progress.emit(mapped, msg)

            def is_cancelled():
                return self._cancel

            scan_results = scanner.scan(
                self.video_path,
                progress_callback=scan_progress,
                cancel_flag=is_cancelled,
            )

            if self._cancel:
                self.progress.emit(0, "已取消")
                return

            # 步骤 C: 匹配段落
            self.progress.emit(90, "正在匹配人脸段落...")
            matcher = Matcher(
                threshold=self.threshold,
                merge_mode=self.merge_mode,
                merge_gap=self.merge_gap,
                fixed_seconds=self.fixed_seconds,
                min_segment_duration=self.min_segment_duration,
            )

            # scan_results 是 ScanResult 对象列表，转成 matcher 需要的格式
            match_input = [(r.timestamp_ms, r.embedding) for r in scan_results]
            segments = matcher.match(query_emb, match_input)

            self.progress.emit(100, f"扫描完成，找到 {len(segments)} 个匹配段落")
            self.finished.emit(segments)

        except Exception as e:
            self.error.emit(str(e))
