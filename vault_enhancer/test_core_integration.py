import os
import unittest
from unittest.mock import patch, MagicMock
from vault_enhancer import core
from vault_enhancer.parakeet_wrapper import TranscriptSegment

class TestCoreIntegration(unittest.TestCase):
    @patch("vault_enhancer.media.isolate_vocals_with_demucs")
    def test_transcribe_video_mocked(self, mock_isolate):
        mock_isolate.return_value = None  # triggers fallback to input_file

        mock_segments = [TranscriptSegment(1, 0.0, 1.0, "Hello world")]

        mock_transcriber = MagicMock()
        mock_transcriber.transcribe_file.return_value = mock_segments

        input_file = "dummy_video.mp4"
        output_file = "dummy_video.srt"

        # Create a dummy file to pass os.path.isfile check
        with open(input_file, "w") as f:
            f.write("dummy content")

        try:
            with patch("vault_enhancer.core.get_parakeet_model", return_value=mock_transcriber):
                outputs = core.transcribe_video(
                    input_file,
                    output_file=output_file,
                    skip_vocal_isolation=True,
                    overwrite=True,
                    engine="parakeet"
                )

            self.assertEqual(len(outputs), 1)
            self.assertTrue(os.path.exists(output_file))

            with open(output_file, "r", encoding="utf-8") as f:
                content = f.read()
            self.assertIn("Hello world", content)

        finally:
            if os.path.exists(input_file):
                os.remove(input_file)
            if os.path.exists(output_file):
                os.remove(output_file)

if __name__ == "__main__":
    unittest.main()
