"""人脸匹配模块 — 嵌入向量相似度比对 + 段落合并"""
import numpy as np


class Matcher:
    """将查询嵌入与视频扫描结果比对，找出匹配的视频段落"""

    def __init__(
        self,
        threshold: float = 0.55,
        merge_mode: str = "auto",
        merge_gap: float = 2.0,
        fixed_seconds: float = 3.0,
        min_segment_duration: float = 1.0,
    ):
        self.threshold = threshold
        self.merge_mode = merge_mode
        self.merge_gap = merge_gap
        self.fixed_seconds = fixed_seconds
        self.min_segment_duration = min_segment_duration

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """计算两个向量的余弦相似度"""
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def match(
        self,
        query_embedding: np.ndarray,
        scan_results: list,
    ) -> list[tuple[int, int, float]]:
        """在扫描结果中查找与查询嵌入匹配的段落。"""
        # 1. 计算每帧相似度，过滤低于阈值的
        matched_frames = []
        for timestamp_ms, embedding in scan_results:
            sim = self.cosine_similarity(query_embedding, embedding)
            if sim >= self.threshold:
                matched_frames.append((timestamp_ms, sim))

        if not matched_frames:
            return []

        matched_frames.sort(key=lambda x: x[0])

        if self.merge_mode == "fixed":
            return self._merge_fixed_length(matched_frames)
        else:
            return self._merge_auto(matched_frames)

    def _merge_auto(self, matched_frames: list) -> list[tuple[int, int, float]]:
        """自动合并相邻匹配帧"""
        gap_ms = self.merge_gap * 1000

        segments = []
        seg_start = matched_frames[0][0]
        seg_end = matched_frames[0][0]
        seg_scores = [matched_frames[0][1]]

        for i in range(1, len(matched_frames)):
            ts, score = matched_frames[i]
            if ts - seg_end <= gap_ms:
                seg_end = ts
                seg_scores.append(score)
            else:
                self._add_segment(segments, seg_start, seg_end, seg_scores)
                seg_start = ts
                seg_end = ts
                seg_scores = [score]

        self._add_segment(segments, seg_start, seg_end, seg_scores)

        return segments

    def _merge_fixed_length(self, matched_frames: list) -> list[tuple[int, int, float]]:
        """固定长度模式：每个匹配帧前后各取固定秒数，合并重叠区间"""
        half_ms = self.fixed_seconds * 1000

        intervals = []
        for ts, score in matched_frames:
            start = max(0, int(ts - half_ms))
            end = int(ts + half_ms)
            intervals.append((start, end, score))

        intervals.sort(key=lambda x: x[0])
        merged = []
        cur_start, cur_end, cur_scores = intervals[0][0], intervals[0][1], [intervals[0][2]]

        for start, end, score in intervals[1:]:
            if start <= cur_end:
                cur_end = max(cur_end, end)
                cur_scores.append(score)
            else:
                self._add_segment(merged, cur_start, cur_end, cur_scores)
                cur_start, cur_end, cur_scores = start, end, [score]

        self._add_segment(merged, cur_start, cur_end, cur_scores)
        return merged

    def _add_segment(self, segments, start_ms, end_ms, scores):
        """添加段落（如果满足最短时长要求）"""
        duration_s = (end_ms - start_ms) / 1000.0
        if duration_s >= self.min_segment_duration:
            avg_score = float(np.mean(scores))
            segments.append((start_ms, end_ms, round(avg_score, 4)))
