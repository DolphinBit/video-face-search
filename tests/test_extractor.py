import json
from pathlib import Path
from src.extractor import SegmentExtractor


class TestSegmentExtractor:
    def test_extract_creates_files(self, test_video_path, output_dir, known_segments):
        """截取段落应生成 mp4 文件和 result.json"""
        extractor = SegmentExtractor()
        results = extractor.extract(test_video_path, known_segments, str(output_dir))

        assert len(results) == 2

        for i, path in enumerate(results):
            p = Path(path)
            assert p.exists()
            assert p.suffix == ".mp4"
            assert p.stat().st_size > 0

        # 检查 result.json
        result_json = output_dir / "result.json"
        assert result_json.exists()
        data = json.loads(result_json.read_text())
        assert len(data) == 2
        assert data[0]["index"] == 1
        assert data[0]["start_ms"] == 1000
        assert data[0]["end_ms"] == 3000

    def test_extract_creates_segments_dir(self, test_video_path, output_dir, known_segments):
        """segments/ 子目录应被自动创建"""
        extractor = SegmentExtractor()
        extractor.extract(test_video_path, known_segments, str(output_dir))
        assert (output_dir / "segments").is_dir()

    def test_extract_empty_segments(self, test_video_path, output_dir):
        """空段落列表应生成空结果"""
        extractor = SegmentExtractor()
        results = extractor.extract(test_video_path, [], str(output_dir))
        assert results == []
