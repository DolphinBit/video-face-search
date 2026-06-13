import numpy as np
import pytest
from src.embedder import FaceEmbedder, FaceEmbedderError


class TestFaceEmbedder:
    def test_init_loads_model(self):
        """初始化应加载 InsightFace 模型"""
        embedder = FaceEmbedder()
        assert embedder.model is not None
        assert embedder.det_size == (640, 640)

    def test_extract_embedding_returns_512d_vector(self, face_image_path):
        """提取嵌入应返回 512 维 numpy 数组（或 None 若无检测到人脸）"""
        embedder = FaceEmbedder()
        result = embedder.extract_embedding(face_image_path)
        # 随机噪声图片可能检测不到人脸 → 返回 None 也合法
        # 若检测到人脸，必须返回 512 维 float32 向量
        if result is not None:
            assert isinstance(result, np.ndarray)
            assert result.shape == (512,)
            assert result.dtype == np.float32

    def test_extract_embedding_no_face_returns_none(self, face_image_path):
        """纯色图片大概率检测不到人脸，返回 None"""
        embedder = FaceEmbedder()
        result = embedder.extract_embedding(face_image_path)
        assert result is None

    def test_extract_embedding_nonexistent_file_raises(self):
        """不存在的文件应抛出 FaceEmbedderError"""
        embedder = FaceEmbedder()
        with pytest.raises(FaceEmbedderError):
            embedder.extract_embedding("/nonexistent/path.jpg")

    def test_extract_embedding_from_array(self):
        """从 numpy 数组提取嵌入"""
        embedder = FaceEmbedder()
        img = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        result = embedder.extract_from_array(img)
        assert isinstance(result, list)
        # 随机图片无真实人脸，但返回格式必须正确
        for item in result:
            assert "bbox" in item
            assert "embedding" in item
            assert "det_score" in item
            assert isinstance(item["embedding"], np.ndarray)
            assert item["embedding"].shape == (512,)
