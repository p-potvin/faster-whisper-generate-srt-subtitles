import os
import unittest
from unittest.mock import patch, MagicMock
from video_transcriber import core

class TestCoreIntegration(unittest.TestCase):
    @patch("video_transcriber.core.WhisperModel")
    @patch("video_transcriber.media.get_media_duration_seconds")
    @patch("video_transcriber.media.extract_audio_to_wav")
    @patch("video_transcriber.media.isolate_vocals_with_demucs")
    def test_transcribe_video_mocked(self, mock_isolate, mock_extract, mock_duration, mock_whisper_class):
        # Setup mocks
        mock_model = MagicMock()
        mock_whisper_class.return_value = mock_model
        
        class MockSegment:
            def __init__(self, id, start, end, text):
                self.id = id
                self.start = start
                self.end = end
                self.text = text
        
        mock_segments = [MockSegment(0, 0, 1.0, "Hello world")]
        mock_info = MagicMock()
        mock_info.language = "en"
        mock_info.language_probability = 0.99
        
        mock_model.transcribe.return_value = (mock_segments, mock_info)
        mock_duration.return_value = 10.0
        
        input_file = "dummy_video.mp4"
        output_file = "dummy_video.srt"
        
        # Create a dummy file to pass os.path.isfile check
        with open(input_file, "w") as f:
            f.write("dummy content")
            
        try:
            # We need to mock get_whisper_model to avoid actually loading the model
            with patch("video_transcriber.core.get_whisper_model", return_value=mock_model):
                outputs = core.transcribe_video(
                    input_file,
                    output_file=output_file,
                    isolate_vocals=False, # Skip complex media processing for now
                    overwrite=True
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
