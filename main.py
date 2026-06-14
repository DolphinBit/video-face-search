"""视频人脸查找工具 — 入口"""
import sys
from PyQt6.QtWidgets import QApplication
from src.gui.i18n import Translator
from src.gui.wizard import FaceSearchWizard


def main():
    app = QApplication(sys.argv)
    translator = Translator("zh")
    app.setApplicationName(translator.t("app_name"))
    wizard = FaceSearchWizard(translator)
    wizard.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
