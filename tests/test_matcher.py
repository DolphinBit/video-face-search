import numpy as np
from src.matcher import Matcher


def _make_embedding(seed: float = 0.5):
    """生成一个512维归一化嵌入向量，用于测试"""
    np.random.seed(int(seed * 1000))
    v = np.random.randn(512).astype(np.float32)
    return v / np.linalg.norm(v)


class TestMatcher:
    def test_cosine_similarity_identical(self):
        """相同向量的余弦相似度应为1.0"""
        a = _make_embedding(1.0)
        score = Matcher.cosine_similarity(a, a)
        assert abs(score - 1.0) < 0.001

    def test_cosine_similarity_orthogonal(self):
        """正交向量的余弦相似度应为0"""
        a = np.array([1.0, 0.0] + [0.0] * 510, dtype=np.float32)
        b = np.array([0.0, 1.0] + [0.0] * 510, dtype=np.float32)
        score = Matcher.cosine_similarity(a, b)
        assert abs(score - 0.0) < 0.001

    def test_match_auto_merge(self):
        """自动合并模式：相邻匹配帧在间隔阈值内应合并为一个段落"""
        query_emb = _make_embedding(1.0)

        scan_results = [
            (0, query_emb),
            (500, query_emb),
            (4000, query_emb),
            (4500, query_emb),
        ]

        matcher = Matcher(threshold=0.5, merge_mode="auto", merge_gap=2.0, min_segment_duration=0.0)
        segments = matcher.match(query_emb, scan_results)

        assert len(segments) == 2
        assert segments[0][0] == 0
        assert segments[0][1] == 500
        assert segments[1][0] == 4000
        assert segments[1][1] == 4500

    def test_match_fixed_length(self):
        """固定长度模式：每个匹配位置前后各截取固定秒数"""
        query_emb = _make_embedding(1.0)

        scan_results = [
            (3000, query_emb),
        ]

        matcher = Matcher(threshold=0.5, merge_mode="fixed", fixed_seconds=2.0)
        segments = matcher.match(query_emb, scan_results)

        assert len(segments) == 1
        assert segments[0][0] == 1000
        assert segments[0][1] == 5000

    def test_match_below_threshold_filtered(self):
        """低于阈值的匹配应被过滤"""
        query_emb = _make_embedding(1.0)
        unrelated_emb = _make_embedding(99.0)

        scan_results = [
            (0, unrelated_emb),
        ]

        matcher = Matcher(threshold=0.7, merge_mode="auto")
        segments = matcher.match(query_emb, scan_results)

        assert len(segments) == 0

    def test_min_segment_duration_filter(self):
        """短于最小时长的段落应被丢弃"""
        query_emb = _make_embedding(1.0)

        scan_results = [
            (5000, query_emb),
        ]

        matcher = Matcher(
            threshold=0.5,
            merge_mode="auto",
            min_segment_duration=1.0,
        )
        segments = matcher.match(query_emb, scan_results)

        assert len(segments) == 0
