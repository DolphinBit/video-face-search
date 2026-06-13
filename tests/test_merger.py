from pathlib import Path
from src.merger import VideoMerger


class TestVideoMerger:
    def test_merge_produces_single_file(self, test_video_path, output_dir):
        """合并应产生单个输出文件"""
        merger = VideoMerger()
        output_path = str(output_dir / "merged.mp4")

        result = merger.merge([test_video_path, test_video_path], output_path)

        assert result == output_path
        p = Path(output_path)
        assert p.exists()
        assert p.stat().st_size > 0

    def test_merge_single_file(self, test_video_path, output_dir):
        """单文件合并应直接输出"""
        merger = VideoMerger()
        output_path = str(output_dir / "merged_single.mp4")

        result = merger.merge([test_video_path], output_path)
        assert result == output_path
        assert Path(output_path).exists()

    def test_merge_empty_list_returns_none(self, output_dir):
        """空列表应返回 None"""
        merger = VideoMerger()
        result = merger.merge([], str(output_dir / "empty.mp4"))
        assert result is None
