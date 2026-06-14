"""分步向导的 4 个页面"""
from pathlib import Path
import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QWizardPage, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QSlider, QDoubleSpinBox, QSpinBox, QComboBox,
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

    def __init__(self, translator):
        super().__init__()
        self.translator = translator
        self.setTitle(self.translator.t("page1_title"))
        self.setSubTitle(self.translator.t("page1_subtitle"))

        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        lang_layout = QHBoxLayout()
        lang_layout.addWidget(QLabel(self.translator.t("language_group")))
        self.language_combo = QComboBox()
        self.language_combo.addItems([
            self.translator.t("language_zh"),
            self.translator.t("language_en"),
        ])
        self.language_combo.setCurrentIndex(0 if self.translator.language == "zh" else 1)
        self.language_combo.currentIndexChanged.connect(self._on_language_changed)
        lang_layout.addWidget(self.language_combo)
        lang_layout.addStretch()
        layout.addLayout(lang_layout)

        # 图片预览
        self.image_label = QLabel(self.translator.t("drag_drop_prompt"))
        self.image_label.setFixedSize(300, 300)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setStyleSheet(
            "border: 2px dashed #888; border-radius: 8px; background: #f5f5f5;"
        )
        layout.addWidget(self.image_label, alignment=Qt.AlignmentFlag.AlignCenter)

        # 选择按钮
        btn_layout = QHBoxLayout()
        self.select_btn = QPushButton(self.translator.t("choose_image"))
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
        self._face_ready = False

    def _on_language_changed(self, index: int):
        language = "zh" if index == 0 else "en"
        wizard = self.wizard()
        if wizard is not None:
            wizard.change_language(language)
    
    #以下是替换
    def _select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.translator.t("choose_face_image_dialog"), "",
            self.translator.t("image_file_filter")
        )
        if not path:
            return
        self._face_path = path
        self._refresh_face_display()   # ✅ 只需这一行，后续所有逻辑都复用

    def update_texts(self):
        self.setTitle(self.translator.t("page1_title"))
        self.setSubTitle(self.translator.t("page1_subtitle"))
        self.language_combo.blockSignals(True)
        self.language_combo.setItemText(0, self.translator.t("language_zh"))
        self.language_combo.setItemText(1, self.translator.t("language_en"))
        self.language_combo.setCurrentIndex(0 if self.translator.language == "zh" else 1)
        self.language_combo.blockSignals(False)
        self.image_label.setText(self.translator.t("drag_drop_prompt"))
        self.select_btn.setText(self.translator.t("choose_image"))

    def isComplete(self) -> bool:
        return self._face_path is not None and self._face_ready

    def face_image_path(self) -> str | None:
        return self._face_path
    #以下是新增
    def _refresh_face_display(self):
        """根据当前 self._face_path 重新加载图片、检测人脸并更新界面"""
        if not self._face_path:
            return
        img = cv2.imread(self._face_path)
        if img is None:
            self.image_label.setText(self.translator.t("drag_drop_prompt"))
            self.status_label.setText(self.translator.t("cannot_read_image"))
            self.status_label.setStyleSheet("color: red;")
            self._face_ready = False
            self.completeChanged.emit()
            return

        # 显示图片预览
        pixmap = _cv_to_qpixmap(img, 280, 280)
        self.image_label.setPixmap(pixmap)

        # 人脸检测
        try:
            if self._embedder is None:
                self._embedder = FaceEmbedder()
            faces = self._embedder.model.get(img)

            if not faces:
                self._face_ready = False
                self.status_label.setText(self.translator.t("no_face_detected"))
                self.status_label.setStyleSheet("color: orange;")
            elif len(faces) == 1:
                self._face_ready = True
                self.status_label.setText(
                    self.translator.t("detected_one_face", score=faces[0].det_score)
                )
                self.status_label.setStyleSheet("color: green;")
            else:
                self._face_ready = True
                best = max(faces, key=lambda f: f.det_score)
                self.status_label.setText(
                    self.translator.t(
                        "detected_many_faces",
                        count=len(faces),
                        score=best.det_score,
                    )
                )
                self.status_label.setStyleSheet("color: orange;")
        except Exception as e:
            self.status_label.setText(self.translator.t("face_detection_failed", error=e))
            self.status_label.setStyleSheet("color: red;")
            self._face_ready = False

        self.completeChanged.emit()

    def _select_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.translator.t("choose_face_image_dialog"), "",
            self.translator.t("image_file_filter")
        )
        if not path:
            return
        self._face_path = path
        self._refresh_face_display()   # 复用统一逻辑

    def update_texts(self):
        self.setTitle(self.translator.t("page1_title"))
        self.setSubTitle(self.translator.t("page1_subtitle"))
        self.language_combo.blockSignals(True)
        self.language_combo.setItemText(0, self.translator.t("language_zh"))
        self.language_combo.setItemText(1, self.translator.t("language_en"))
        self.language_combo.setCurrentIndex(0 if self.translator.language == "zh" else 1)
        self.language_combo.blockSignals(False)
        # 关键修改：根据是否有图片决定显示内容
        if self._face_path:
            self._refresh_face_display()   # 重新加载并显示（同时刷新文字）
        else:
            self.image_label.setText(self.translator.t("drag_drop_prompt"))
        self.select_btn.setText(self.translator.t("choose_image"))

class Step2VideoPage(QWizardPage):
    """步骤2：选择视频文件"""

    def __init__(self, translator):
        super().__init__()
        self.translator = translator
        self.setTitle(self.translator.t("page2_title"))
        self.setSubTitle(self.translator.t("page2_subtitle"))

        layout = QVBoxLayout()

        # 视频文件选择
        file_layout = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setPlaceholderText(self.translator.t("video_placeholder"))
        file_layout.addWidget(self.path_edit)
        self.browse_btn = QPushButton(self.translator.t("browse"))
        self.browse_btn.clicked.connect(self._select_video)
        file_layout.addWidget(self.browse_btn)
        layout.addLayout(file_layout)

        # 视频信息
        self.info_label = QLabel("")
        layout.addWidget(self.info_label)

        # 帧采样间隔
        interval_layout = QHBoxLayout()
        self.frame_label = QLabel(self.translator.t("frame_interval_label"))
        interval_layout.addWidget(self.frame_label)
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 30)
        self.interval_spin.setValue(5)
        self.interval_spin.setSuffix(self.translator.t("seconds"))
        self.interval_spin.setToolTip(self.translator.t("frame_interval_tooltip"))
        interval_layout.addWidget(self.interval_spin)
        interval_layout.addStretch()
        layout.addLayout(interval_layout)

        layout.addStretch()
        self.setLayout(layout)
        self._video_path: str | None = None

    def _select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.translator.t("choose_video_dialog"), "",
            self.translator.t("video_file_filter")
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
                self.translator.t(
                    "video_info_template",
                    width=width,
                    height=height,
                    fps=fps,
                    frames=total_frames,
                    duration=duration_s,
                    estimated=est_frames,
                )
            )
            cap.release()

        self.completeChanged.emit()

    def update_texts(self):
        self.setTitle(self.translator.t("page2_title"))
        self.setSubTitle(self.translator.t("page2_subtitle"))
        self.path_edit.setPlaceholderText(self.translator.t("video_placeholder"))
        self.browse_btn.setText(self.translator.t("browse"))
        self.frame_label.setText(self.translator.t("frame_interval_label"))
        self.interval_spin.setSuffix(self.translator.t("seconds"))
        self.interval_spin.setToolTip(self.translator.t("frame_interval_tooltip"))

    def isComplete(self) -> bool:
        return self._video_path is not None

    def video_path(self) -> str | None:
        return self._video_path

    def frame_interval(self) -> int:
        return self.interval_spin.value()
    #以下是新增
    def _refresh_video_info(self):
        """根据当前 _video_path 重新生成视频信息（用于语言切换后更新）"""
        if not self._video_path:
            return
        cap = cv2.VideoCapture(self._video_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration_s = total_frames / fps if fps > 0 else 0
            interval = self.interval_spin.value()
            est_frames = total_frames // interval if interval > 0 else total_frames
            self.info_label.setText(
                self.translator.t(
                    "video_info_template",
                    width=width,
                    height=height,
                    fps=fps,
                    frames=total_frames,
                    duration=duration_s,
                    estimated=est_frames,
                )
            )
            cap.release()
        else:
            self.info_label.setText(self.translator.t("cannot_read_video"))

    def _select_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, self.translator.t("choose_video_dialog"), "",
            self.translator.t("video_file_filter")
        )
        if not path:
            return
        self._video_path = path
        self.path_edit.setText(path)
        self._refresh_video_info()   # 替换原来的重复代码
        self.completeChanged.emit()

    def update_texts(self):
        self.setTitle(self.translator.t("page2_title"))
        self.setSubTitle(self.translator.t("page2_subtitle"))
        self.path_edit.setPlaceholderText(self.translator.t("video_placeholder"))
        self.browse_btn.setText(self.translator.t("browse"))
        self.frame_label.setText(self.translator.t("frame_interval_label"))
        self.interval_spin.setSuffix(self.translator.t("seconds"))
        self.interval_spin.setToolTip(self.translator.t("frame_interval_tooltip"))
        # 关键：刷新视频信息（如果已选择视频）
        self._refresh_video_info()

class Step3MatchPage(QWizardPage):
    """步骤3：匹配设置与执行扫描"""

    def __init__(self, translator):
        super().__init__()
        self.translator = translator
        self.setTitle(self.translator.t("page3_title"))
        self.setSubTitle(self.translator.t("page3_subtitle"))

        layout = QVBoxLayout()

        # --- 相似度阈值 ---
        self.threshold_group = QGroupBox(self.translator.t("similarity_threshold"))
        threshold_layout = QVBoxLayout()
        header_layout = QHBoxLayout()
        self.loose_label = QLabel(self.translator.t("loose"))
        header_layout.addWidget(self.loose_label)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(30, 90)
        self.threshold_slider.setValue(55)
        self.threshold_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.threshold_slider.setTickInterval(10)
        header_layout.addWidget(self.threshold_slider)
        self.strict_label = QLabel(self.translator.t("strict"))
        header_layout.addWidget(self.strict_label)
        threshold_layout.addLayout(header_layout)
        self.threshold_label = QLabel(self.translator.t("current_threshold", value=0.55))
        self.threshold_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.threshold_slider.valueChanged.connect(
            lambda v: self.threshold_label.setText(
                self.translator.t("current_threshold", value=v / 100)
            )
        )
        threshold_layout.addWidget(self.threshold_label)
        self.threshold_group.setLayout(threshold_layout)
        layout.addWidget(self.threshold_group)

        # --- 合并模式 ---
        self.merge_group = QGroupBox(self.translator.t("merge_mode_group"))
        merge_layout = QVBoxLayout()

        self.auto_radio = QRadioButton(self.translator.t("auto_merge"))
        self.auto_radio.setChecked(True)
        merge_layout.addWidget(self.auto_radio)

        auto_sub = QHBoxLayout()
        self.gap_label = QLabel(self.translator.t("interval_threshold"))
        auto_sub.addWidget(self.gap_label)
        self.gap_spin = QDoubleSpinBox()
        self.gap_spin.setRange(0.5, 10.0)
        self.gap_spin.setValue(2.0)
        self.gap_spin.setSuffix(self.translator.t("seconds"))
        self.gap_spin.setSingleStep(0.5)
        auto_sub.addWidget(self.gap_spin)
        auto_sub.addStretch()
        merge_layout.addLayout(auto_sub)

        self.fixed_radio = QRadioButton(self.translator.t("fixed_length"))
        merge_layout.addWidget(self.fixed_radio)

        fixed_sub = QHBoxLayout()
        self.fixed_label = QLabel(self.translator.t("before_after"))
        fixed_sub.addWidget(self.fixed_label)
        self.fixed_spin = QDoubleSpinBox()
        self.fixed_spin.setRange(1.0, 30.0)
        self.fixed_spin.setValue(3.0)
        self.fixed_spin.setSuffix(self.translator.t("seconds"))
        self.fixed_spin.setSingleStep(0.5)
        self.fixed_spin.setEnabled(False)
        fixed_sub.addWidget(self.fixed_spin)
        fixed_sub.addStretch()
        merge_layout.addLayout(fixed_sub)

        self.auto_radio.toggled.connect(
            lambda checked: self.fixed_spin.setEnabled(not checked)
        )

        self.merge_group.setLayout(merge_layout)
        layout.addWidget(self.merge_group)

        # --- 最短段落时长 ---
        min_layout = QHBoxLayout()
        self.min_label = QLabel(self.translator.t("min_segment_duration"))
        min_layout.addWidget(self.min_label)
        self.min_dur_spin = QDoubleSpinBox()
        self.min_dur_spin.setRange(0.5, 10.0)
        self.min_dur_spin.setValue(1.0)
        self.min_dur_spin.setSuffix(self.translator.t("seconds"))
        min_layout.addWidget(self.min_dur_spin)
        min_layout.addStretch()
        layout.addLayout(min_layout)

        # --- 扫描按钮 ---
        btn_layout = QHBoxLayout()
        self.scan_btn = QPushButton(self.translator.t("start_scan"))
        self.scan_btn.setMinimumHeight(40)
        self.scan_btn.clicked.connect(self._start_scan)
        btn_layout.addWidget(self.scan_btn)
        self.cancel_btn = QPushButton(self.translator.t("cancel"))
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self._cancel_scan)
        btn_layout.addWidget(self.cancel_btn)
        layout.addLayout(btn_layout)

        # --- 进度条 ---
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        self.progress_label = QLabel(self.translator.t("waiting_start"))
        layout.addWidget(self.progress_label)

        layout.addStretch()
        self.setLayout(layout)

        self._worker: ScanWorker | None = None
        self._segments: list = []

    def update_texts(self):
        self.setTitle(self.translator.t("page3_title"))
        self.setSubTitle(self.translator.t("page3_subtitle"))
        self.threshold_group.setTitle(self.translator.t("similarity_threshold"))
        self.loose_label.setText(self.translator.t("loose"))
        self.strict_label.setText(self.translator.t("strict"))
        self.threshold_label.setText(self.translator.t(
            "current_threshold", value=self.threshold_slider.value() / 100
        ))
        self.merge_group.setTitle(self.translator.t("merge_mode_group"))
        self.auto_radio.setText(self.translator.t("auto_merge"))
        self.gap_label.setText(self.translator.t("interval_threshold"))
        self.gap_spin.setSuffix(self.translator.t("seconds"))
        self.fixed_radio.setText(self.translator.t("fixed_length"))
        self.fixed_label.setText(self.translator.t("before_after"))
        self.min_label.setText(self.translator.t("min_segment_duration"))
        self.min_dur_spin.setSuffix(self.translator.t("seconds"))
        self.scan_btn.setText(self.translator.t("start_scan"))
        self.cancel_btn.setText(self.translator.t("cancel"))
        self.progress_label.setText(self.translator.t("waiting_start"))

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
                self, self.translator.t("scan_result_title"),
                self.translator.t("no_matches_found")
            )
        self.completeChanged.emit()

    def _on_error(self, msg: str):
        self.scan_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_label.setText(self.translator.t("scan_error", error=msg))
        QMessageBox.critical(self, self.translator.t("scan_error_title"), msg)

    def isComplete(self) -> bool:
        return len(self._segments) > 0

    def segments(self) -> list:
        return self._segments


class Step4ExportPage(QWizardPage):
    """步骤4：查看结果并导出"""

    def __init__(self, translator):
        super().__init__()
        self.translator = translator
        self.setTitle(self.translator.t("page4_title"))
        self.setSubTitle(self.translator.t("page4_subtitle"))

        layout = QVBoxLayout()

        # 段落列表
        self.list_widget = QListWidget()
        self.list_widget.setSelectionMode(
            QListWidget.SelectionMode.MultiSelection
        )
        self.segment_label = QLabel(self.translator.t("match_segments"))
        layout.addWidget(self.segment_label)
        layout.addWidget(self.list_widget)

        # 输出目录
        dir_layout = QHBoxLayout()
        self.output_dir_label = QLabel(self.translator.t("output_dir"))
        dir_layout.addWidget(self.output_dir_label)
        self.output_edit = QLineEdit()
        self.output_edit.setReadOnly(True)
        dir_layout.addWidget(self.output_edit)
        self.dir_btn = QPushButton(self.translator.t("select_dir"))
        self.dir_btn.clicked.connect(self._select_dir)
        dir_layout.addWidget(self.dir_btn)
        layout.addLayout(dir_layout)

        # 导出按钮
        btn_layout = QHBoxLayout()
        self.export_segments_btn = QPushButton(self.translator.t("export_selected"))
        self.export_segments_btn.setMinimumHeight(36)
        self.export_segments_btn.clicked.connect(self._export_segments)
        btn_layout.addWidget(self.export_segments_btn)

        self.export_merged_btn = QPushButton(self.translator.t("export_merged"))
        self.export_merged_btn.setMinimumHeight(36)
        self.export_merged_btn.clicked.connect(self._export_merged)
        btn_layout.addWidget(self.export_merged_btn)
        layout.addLayout(btn_layout)

        self.setLayout(layout)
        self._segments: list = []
        self._output_dir: str = ""

    def update_texts(self):
        self.setTitle(self.translator.t("page4_title"))
        self.setSubTitle(self.translator.t("page4_subtitle"))
        self.segment_label.setText(self.translator.t("match_segments"))
        self.output_dir_label.setText(self.translator.t("output_dir"))
        self.dir_btn.setText(self.translator.t("select_dir"))
        self.export_segments_btn.setText(self.translator.t("export_selected"))
        self.export_merged_btn.setText(self.translator.t("export_merged"))

    def _load_segments(self, segments: list):
        """从扫描结果加载段落列表"""
        self._segments = segments
        self.list_widget.clear()
        for i, (start_ms, end_ms, score) in enumerate(segments, start=1):
            duration_s = (end_ms - start_ms) / 1000.0
            item = QListWidgetItem(
                self.translator.t(
                    "segment_item",
                    index=i,
                    start=start_ms / 1000,
                    end=end_ms / 1000,
                    duration=duration_s,
                    score=score,
                )
            )
            item.setData(Qt.ItemDataRole.UserRole, i - 1)
            self.list_widget.addItem(item)

    def _select_dir(self):
        d = QFileDialog.getExistingDirectory(self, self.translator.t("select_output_dir_dialog"))
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
            QMessageBox.warning(self, self.translator.t("warning_title"), self.translator.t("no_exportable_segments"))
            return

        selected = self._get_selected_segments()
        sel_segments = [self._segments[i] for i in selected]

        from src.extractor import SegmentExtractor
        extractor = SegmentExtractor()
        video_path = self.wizard().page(1).video_path()

        try:
            paths = extractor.extract(video_path, sel_segments, self._output_dir)
            QMessageBox.information(
                self, self.translator.t("warning_title"),
                self.translator.t("export_complete", count=len(paths), path=f"{self._output_dir}/segments/")
            )
        except Exception as e:
            QMessageBox.critical(self, self.translator.t("scan_error_title"), self.translator.t("export_failed", error=str(e)))

    def _export_merged(self):
        if not self._segments or not self._output_dir:
            QMessageBox.warning(self, self.translator.t("warning_title"), self.translator.t("no_exportable_segments"))
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
                self, self.translator.t("warning_title"),
                self.translator.t("merged_export_complete", path=merged_path)
            )
        except Exception as e:
            QMessageBox.critical(self, self.translator.t("scan_error_title"), self.translator.t("export_failed", error=str(e)))
    # 以下是新增
    def _load_segments(self, segments: list):
        """从扫描结果加载段落列表（原逻辑）"""
        self._segments = segments
        self._refresh_list()   # 抽取出刷新列表的逻辑

    def _refresh_list(self):
        """仅用当前的 self._segments 和当前语言重新生成列表项"""
        self.list_widget.clear()
        if not self._segments:
            return
        for i, (start_ms, end_ms, score) in enumerate(self._segments, start=1):
            duration_s = (end_ms - start_ms) / 1000.0
            item = QListWidgetItem(
                self.translator.t(
                    "segment_item",
                    index=i,
                    start=start_ms / 1000,
                    end=end_ms / 1000,
                    duration=duration_s,
                    score=score,
                )
            )
            item.setData(Qt.ItemDataRole.UserRole, i - 1)
            self.list_widget.addItem(item)

    def update_texts(self):
        self.setTitle(self.translator.t("page4_title"))
        self.setSubTitle(self.translator.t("page4_subtitle"))
        self.segment_label.setText(self.translator.t("match_segments"))
        self.output_dir_label.setText(self.translator.t("output_dir"))
        self.dir_btn.setText(self.translator.t("select_dir"))
        self.export_segments_btn.setText(self.translator.t("export_selected"))
        self.export_merged_btn.setText(self.translator.t("export_merged"))
        # 关键：刷新列表项的文字为当前语言
        self._refresh_list()