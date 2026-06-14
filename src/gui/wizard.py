"""主向导窗口 — 组合 4 个步骤页面"""
from PyQt6.QtWidgets import QWizard
from PyQt6.QtGui import QFont
from src.gui.i18n import Translator
from src.gui.pages import (
    Step1FacePage,
    Step2VideoPage,
    Step3MatchPage,
    Step4ExportPage,
)


class FaceSearchWizard(QWizard):
    """视频人脸查找 — 分步向导"""

    def __init__(self, translator: Translator):
        super().__init__()
        self.translator = translator
        self.setWindowTitle(self.translator.t("window_title"))
        self.setMinimumSize(600, 550)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)

        # 全局字体
        font = QFont("Microsoft YaHei", 10)
        self.setFont(font)

        # 添加页面
        self.page1 = Step1FacePage(self.translator)
        self.page2 = Step2VideoPage(self.translator)
        self.page3 = Step3MatchPage(self.translator)
        self.page4 = Step4ExportPage(self.translator)

        self.addPage(self.page1)
        self.addPage(self.page2)
        self.addPage(self.page3)
        self.addPage(self.page4)

        self.setStyleSheet("""
            QWizard {
                background: #fafafa;
            }
            QPushButton {
                padding: 6px 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #fff;
            }
            QPushButton:hover {
                background: #e8e8e8;
            }
            QPushButton:disabled {
                color: #aaa;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #4a90d9;
                border-radius: 3px;
            }
        """)

    def change_language(self, language: str):
        self.translator.set_language(language)
        self.setWindowTitle(self.translator.t("window_title"))
        for page in (self.page1, self.page2, self.page3, self.page4):
            page.update_texts()

        # 样式
        self.setStyleSheet("""
            QWizard {
                background: #fafafa;
            }
            QPushButton {
                padding: 6px 16px;
                border: 1px solid #ccc;
                border-radius: 4px;
                background: #fff;
            }
            QPushButton:hover {
                background: #e8e8e8;
            }
            QPushButton:disabled {
                color: #aaa;
            }
            QProgressBar {
                border: 1px solid #ccc;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background: #4a90d9;
                border-radius: 3px;
            }
        """)
