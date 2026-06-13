"""视频人脸查找工具 — 入口"""
import sys
from PyQt6.QtWidgets import QApplication
from src.gui.wizard import FaceSearchWizard


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("视频人脸查找")
    wizard = FaceSearchWizard()
    wizard.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
