import tempfile
import subprocess
import shutil
from pathlib import Path
import pytest
import numpy as np
import cv2


@pytest.fixture
def project_root():
    """项目根目录"""
    return Path(__file__).parent.parent


@pytest.fixture
def output_dir():
    """临时输出目录，测试后自动清理"""
    d = tempfile.mkdtemp(prefix="vfstest_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def face_image_path(output_dir):
    """生成一张含有人脸区域（纯色块模拟）的测试图片"""
    img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    path = str(output_dir / "test_face.jpg")
    cv2.imwrite(path, img)
    return path


@pytest.fixture
def test_video_path(output_dir):
    """用 ffmpeg 生成一段 3 秒纯色测试视频（mp4）"""
    path = output_dir / "test_video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=blue:size=320x240:duration=3:rate=10",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(path),
        ],
        capture_output=True,
        check=True,
    )
    return str(path)


@pytest.fixture
def known_segments():
    """模拟匹配段落（毫秒单位）"""
    return [
        (1000, 3000, 0.72),
        (5000, 8000, 0.85),
    ]
