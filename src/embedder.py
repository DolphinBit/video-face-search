"""人脸嵌入提取模块 — 基于 InsightFace"""
import numpy as np
from insightface.app import FaceAnalysis


class FaceEmbedderError(Exception):
    """嵌入提取异常"""
    pass


class FaceEmbedder:
    """使用 InsightFace 提取人脸嵌入向量（512维）"""

    def __init__(self, det_size: tuple = (640, 640)):
        self.det_size = det_size
        self.model = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.model.prepare(ctx_id=0, det_size=det_size)

    def extract_embedding(self, image_path: str) -> np.ndarray | None:
        """从图片文件提取人脸嵌入向量。

        Args:
            image_path: 图片文件路径

        Returns:
            512维嵌入向量，如果未检测到人脸则返回 None

        Raises:
            FaceEmbedderError: 文件不存在或读取失败
        """
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            raise FaceEmbedderError(f"无法读取图片: {image_path}")

        faces = self.model.get(img)
        if not faces:
            return None

        best = max(faces, key=lambda f: f.det_score)
        return best.embedding.copy()

    def extract_from_array(self, img: np.ndarray) -> list:
        """从 numpy 图像数组提取所有人脸的嵌入向量。

        Args:
            img: BGR 格式的 numpy 数组 (H, W, 3)

        Returns:
            [{"bbox", "embedding", "det_score"}, ...] 列表
        """
        faces = self.model.get(img)
        results = []
        for f in faces:
            results.append({
                "bbox": f.bbox.astype(np.int32).tolist(),
                "embedding": f.embedding.copy(),
                "det_score": float(f.det_score),
            })
        return results
