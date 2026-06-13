"""分步向导的 4 个页面"""
from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWizardPage, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSlider, QDoubleSpinBox, QSpinBox,
    QListWidget, QListWidgetItem, QProgressBar, QGroupBox, QRadioButton,
    QMessageBox, QLineEdit,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QImage
from src.embedder import FaceEmbedder
from src.gui.worker import ScanWorker


def _cv_to_qpixmap(cv_img_bgr, w=200, h=200):
    """OpenCV BGR 图像转 QPixmap"""
    rgb = cv2.cvtColor(cv_img_bgr, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = rgb.shape
    qimg = QImage(rgb.data, w_img, h_img, w_img * 3, QImage.Format.Format_RGB888)
    pixmap = QPixmap.fromImage(qimg)
    return pixmap.scaled(w, h, Qt.AspectRatioMode.KeepAspectRatio)


class Step1FacePage(QWizardPage):
    """步骤1：上传人脸图片"""

    def __init__(self):
        super().__init__()
        self.setTitle("步骤 1：上传人脸图片")
        self.setSubTitle("选择一张包含人脸的图片作为查询依据")

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # 图片预览
        self.image_label = QLabel("拖拽或点击下方按钮选择图片")
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 2px dashed #888; border-radius: 8px; background: #f5f5f5;"
        )
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 选择按钮
        btn_layout = QHBoxLayout()
        self.select_btn = QPushButton("选择图片")
        self.select_btn.clicked.connect(self._select_image)
        btn_layout.addWidget(self.select_btn)
        layout.addLayout(btn_layout)

        # 状态提示
        self.status_label = QLabel("")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)

        self.setLayout(layout)
        self._face_path: str | None = None
        self._embedder: FaceEmbedder | None = None

    def _select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择人脸图片", "",
            "图片文件 (*.jpg *.jpeg *.png *.bmp);;所有文件 (*)"
        )
        if not path:
            return

        self._face_path = path
        img = cv2.imread(path)
        if img is None:
            self.status_label.setText("⚠️ 无法读取图片")
            self.status_label.setStyleSheet("color: red;")
            return

        # 显示预览
        pixmap = _cv_to_qpixmap(img, 280, 280)
        self.image_label.setPixmap(pixmap)

        # 检测人脸
        try:
            if self._embedder is None:
                self._embedder = FaceEmbedder()
            faces = self._embedder.model.get(img)

            if not faces:
                self.status_label.setText("⚠️ 未检测到人脸，请换一张图片")
                self.status_label.setStyleSheet("color: orange;")
            elif len(faces) == 1:
                self.status_label.setText(
                    f"✅ 检测到 1 张人脸 (置信度: {faces[0].det_score:.2f})"
                )
                self.status_label.setStyleSheet("color: green;")
            else:
                best = max(faces, key=lambda f: f.det_score)
                self.status_label.setText(
                    f"⚠️ 检测到 {len(faces)} 张人脸，已自动选择置信度最高的 (置信度: {best.det_score:.2f})"
                )
                self.status_label.setStyleSheet("color: orange;")

            self.completeChanged.emit()
        except Exception as e:
            self.status_label.setText(f"❌ 人脸检测失败: {e}")
            self.status_label.setStyleSheet("color: red;")

    def isComplete(self) -> bool:
        return self._face_path is not None and (
            self.status_label.text().startswith("✅")
            or "已自动选择" in self.status_label.text()
        )

    def face_image_path(self) -> str | None:
        return self._face_path


class Step2VideoPage(QWizardPage):
    """步骤2：选择视频文件"""

    def __init__(self):
        super().__init__()
        self.setTitle("步骤 2：选择视频文件")
        self.setSubTitle("选择要搜索的视频，设置采样参数")

        layout = QVBoxLayout()

        # 视频文件选择
        file_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText("未选择视频文件...")
        file_layout.addWidget(self.path_edit)
        self.browse_btn = QPushButton("浏览...")
        self.browse_btn.clicked.connect(self._select_video)
        file_layout.addWidget(self.browse_btn)
        layout.addLayout(file_layout)

        # 视频信息
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

        # 帧采样间隔
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("帧采样间隔："))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 30)
        self.interval_spin.setValue(5)
        self.interval_spin.setSuffix(" 帧")
        self.interval_spin.setToolTip("每隔多少帧检测一次人脸，值越小越精确但越慢")
        interval_layout.addWidget(self.interval_spin)
        interval_layout.addStretch()
        layout.addLayout(interval_layout)

        layout.addStretch()
        self.setLayout(layout)
        self._video_path: str | None = None

    def _select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mkv *.mov);;所有文件 (*)"
        )
        if not path:
            return

        self._video_path = path
        self.path_edit.setText(path)

        cap = cv2.VideoCapture(path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_s = total_frames / fps if fps > 0 else 0

            interval = self.interval_spin.value()
            est_frames = total_frames // interval if interval > 0 else total_frames

            self.info_label.setText(
                f"分辨率: {width}×{height} | 帧率: {fps:.1f} fps | "
                f"总帧数: {total_frames} | 时长: {duration_s:.1f} 秒\n"
                f"预计采样: ~{est_frames} 帧"
            )
            cap.release()

        self.completeChanged.emit()

    def isComplete(self) -> bool:
        return self._video_path is not None

    def video_path(self) -> str | None:
        return self._video_path

    def frame_interval(self) -> int:
        return self.interval_spin.value()


class Step3MatchPage(QWizardPage):
    """步骤3：匹配设置与执行扫描"""

    def __init__(self):
        super().__init__()
        self.setTitle("步骤 3：匹配设置并扫描")
        self.setSubTitle("调整匹配参数，然后开始扫描视频")

        layout = QVBoxLayout()

        # --- 相似度阈值 ---
        threshold_group = QGroupBox("相似度阈值")
        threshold_layout = QVBoxLayout()
        header_layout = QHBoxLayout()
        header_layout.addWidget(QLabel("宽松"))
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 90)
        self.threshold_slider.setValue(55)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        header_layout.addWidget(self.threshold_slider)
        header_layout.addWidget(QLabel("严格"))
        threshold_layout.addLayout(header_layout)
        self.threshold_label = QLabel("当前阈值: 0.55")
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(f"当前阈值: {v / 100:.2f}")
        )
        threshold_layout.addWidget(self.threshold_label)
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)

        # --- 合并模式 ---
        merge_group = QGroupBox("段落合并模式")
        merge_layout = QVBoxLayout()

        self.auto_radio = QRadioButton("自动合并")
        self.auto_radio.setChecked(True)
        merge_layout.addWidget(self.auto_radio)

        auto_sub = QHBoxLayout()
        auto_sub.addWidget(QLabel("  间隔阈值："))
        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.5, 10.0)
        self.gap_spin.setValue(2.0)
        self.gap_spin.setSuffix(" 秒")
        self.gap_spin.setSingleStep(0.5)
        auto_sub.addWidget(self.gap_spin)
        auto_sub.addStretch()
        merge_layout.addLayout(auto_sub)

        self.fixed_radio = QRadioButton("固定长度")
        merge_layout.addWidget(self.fixed_radio)

        fixed_sub = QHBoxLayout()
        fixed_sub.addWidget(QLabel("  前后各："))
        self.fixed_spin = QDoubleSpinBox()
        self.fixed_spin.setRange(1.0, 30.0)
        self.fixed_spin.setValue(3.0)
        self.fixed_spin.setSuffix(" 秒")
        self.fixed_spin.setSingleStep(0.5)
        self.fixed_spin.setEnabled(False)
        fixed_sub.addWidget(self.fixed_spin)
        fixed_sub.addStretch()
        merge_layout.addLayout(fixed_sub)

        self.auto_radio.toggled.connect(
            lambda checked: self.fixed_spin.setEnabled(not checked)
        )

        merge_group.setLayout(merge_layout)
        layout.addWidget(merge_group)

        # --- 最短段落时长 ---
        min_layout = QHBoxLayout()
        min_layout.addWidget(QLabel("最短段落时长："))
        self.min_dur_spin = QDoubleSpinBox()
        self.min_dur_spin.setRange(0.5, 10.0)
        self.min_dur_spin.setValue(1.0)
        self.min_dur_spin.setSuffix(" 秒")
        min_layout.addWidget(self.min_dur_spin)
        min_layout.addStretch()
        layout.addLayout(min_layout)

        # --- 扫描按钮 ---
        btn_layout = QHBoxLayout()
        self.scan_btn = QPushButton("🔍 开始扫描")
        self.scan_btn.setMinimumHeight(40)
        self.scan_btn.clicked.connect(self._start_scan)
        btn_layout.addWidget(self.scan_btn)
        self.cancel_btn = QPushButton("取消")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_scan)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        # --- 进度条 ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("等待开始...")
        layout.addWidget(self.progress_label)

        layout.addStretch()
        self.setLayout(layout)

        self._worker: ScanWorker | None = None
        self._segments: list = []

    def _start_scan(self):
        wizard = self.wizard()
        page1 = wizard.page(0)
        page2 = wizard.page(1)

        face_path = page1.face_image_path()
        video_path = page2.video_path()
        frame_interval = page2.frame_interval()

        if not face_path or not video_path:
            return

        threshold = self.threshold_slider.value() / 100.0
        merge_mode = "auto" if self.auto_radio.isChecked() else "fixed"

        self._worker = ScanWorker(
            face_image_path=face_path,
            video_path=video_path,
            frame_interval=frame_interval,
            threshold=threshold,
            merge_mode=merge_mode,
            merge_gap=self.gap_spin.value(),
            fixed_seconds=self.fixed_spin.value(),
            min_segment_duration=self.min_dur_spin.value(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)

        self.scan_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self._worker.start()

    def _cancel_scan(self):
        if self._worker and self._worker.isRunning():
            self._worker.cancel()
            self._worker.wait(2000)
        self.scan_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def _on_progress(self, pct: int, msg: str):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(msg)

    def _on_finished(self, segments: list):
        self._segments = segments
        self.scan_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        page4 = self.wizard().page(3)
        page4._load_segments(segments)

        if not segments:
            QMessageBox.information(
                self, "扫描结果",
                "未找到匹配的人脸段落。\n建议降低相似度阈值后重试。"
            )
        self.completeChanged.emit()

    def _on_error(self, msg: str):
        self.scan_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText(f"错误: {msg}")
        QMessageBox.critical(self, "扫描错误", msg)

    def isComplete(self) -> bool:
        return len(self._segments) > 0

    def segments(self) -> list:
        return self._segments


class Step4ExportPage(QWizardPage):
    """步骤4：查看结果并导出"""

    def __init__(self):
        super().__init__()
        self.setTitle("步骤 4：查看结果并导出")
        self.setSubTitle("预览匹配段落，选择要导出的内容")

        layout = QVBoxLayout()

        # 段落列表
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        layout.addWidget(QLabel("匹配段落（可多选）："))
        layout.addWidget(self.list_widget)

        # 输出目录
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(QLabel("输出目录："))
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        dir_layout.addWidget(self.output_edit)
        self.dir_btn = QPushButton("选择...")
        self.dir_btn.clicked.connect(self._select_dir)
        dir_layout.addWidget(self.dir_btn)
        layout.addLayout(dir_layout)

        # 导出按钮
        btn_layout = QHBoxLayout()
        self.export_segments_btn = QPushButton("💾 导出选中段落")
        self.export_segments_btn.setMinimumHeight(36)
        self.export_segments_btn.clicked.connect(self._export_segments)
        btn_layout.addWidget(self.export_segments_btn)

        self.export_merged_btn = QPushButton("🎬 导出合并视频")
        self.export_merged_btn.setMinimumHeight(36)
        self.export_merged_btn.clicked.connect(self._export_merged)
        btn_layout.addWidget(self.export_merged_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self._segments: list = []
        self._output_dir: str = ""

    def _load_segments(self, segments: list):
        """从扫描结果加载段落列表"""
        self._segments = segments
        self.list_widget.clear()
        for i, (start_ms, end_ms, score) in enumerate(segments, start=1):
            duration_s = (end_ms - start_ms) / 1000.0
            item = QListWidgetItem(
                f"段落 {i}: {start_ms/1000:.1f}s → {end_ms/1000:.1f}s "
                f"(时长 {duration_s:.1f}s, 相似度 {score:.2f})"
            )
            item.setData(Qt.ItemDataRole.UserRole, i - 1)
            self.list_widget.addItem(item)

    def _select_dir(self):
        d = QFileDialog.getExistingDirectory(self, "选择输出目录")
        if d:
            self._output_dir = d
            self.output_edit.setText(d)

    def _get_selected_segments(self):
        selected_indices = []
        for item in self.list_widget.selectedItems():
            selected_indices.append(item.data(Qt.ItemDataRole.UserRole))
        if not selected_indices:
            # 无选中 → 全部导出
            return list(range(len(self._segments)))
        return sorted(selected_indices)

    def _export_segments(self):
        if not self._segments or not self._output_dir:
            QMessageBox.warning(self, "提示", "没有可导出的段落或未选择输出目录")
            return

        selected = self._get_selected_segments()
        sel_segments = [self._segments[i] for i in selected]

        from src.extractor import SegmentExtractor
        extractor = SegmentExtractor()
        video_path = self.wizard().page(1).video_path()

        try:
            paths = extractor.extract(video_path, sel_segments, self._output_dir)
            QMessageBox.information(
                self, "导出完成",
                f"已导出 {len(paths)} 个段落到:\n{self._output_dir}/segments/"
            )
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))

    def _export_merged(self):
        if not self._segments or not self._output_dir:
            QMessageBox.warning(self, "提示", "没有可导出的段落或未选择输出目录")
            return

        selected = self._get_selected_segments()
        sel_segments = [self._segments[i] for i in selected]

        from src.extractor import SegmentExtractor
        from src.merger import VideoMerger

        video_path = self.wizard().page(1).video_path()

        try:
            extractor = SegmentExtractor()
            segment_paths = extractor.extract(video_path, sel_segments, self._output_dir)

            merger = VideoMerger()
            merged_path = str(Path(self._output_dir) / "merged.mp4")
            merger.merge(segment_paths, merged_path)

            QMessageBox.information(
                self, "导出完成",
                f"已导出合并视频到:\n{merged_path}"
            )
        except Exception as e:
            QMessageBox.critical(self, "导出失败", str(e))
