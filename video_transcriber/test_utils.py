import os
import unittest
from video_transcriber import utils

class TestUtils(unittest.TestCase):
    def test_format_time(self):
        self.assertEqual(utils.format_time(0), "00:00:00,000")
        self.assertEqual(utils.format_time(1), "00:00:01,000")
        self.assertEqual(utils.format_time(61), "00:01:01,000")
        self.assertEqual(utils.format_time(3661), "01:01:01,000")
        self.assertEqual(utils.format_time(0.5), "00:00:00,500")
        self.assertEqual(utils.format_time(3600.001), "01:00:00,001")

    def test_write_srt(self):
        class MockSegment:
            def __init__(self, id, start, end):
                self.id = id
                self.start = start
                self.end = end
        
        segments = [
            MockSegment(0, 0, 1.5),
            MockSegment(1, 1.5, 3.0)
        ]
        texts = ["Hello", "World"]
        output_path = "test_output.srt"
        
        try:
            utils.write_srt(output_path, segments, texts)
            self.assertTrue(os.path.exists(output_path))
            
            with open(output_path, "r", encoding="utf-8") as f:
                content = f.read()
            
            expected = "1\n00:00:00,000 --> 00:00:01,500\nHello\n\n2\n00:00:01,500 --> 00:00:03,000\nWorld\n\n"
            self.assertEqual(content, expected)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

if __name__ == "__main__":
    unittest.main()
