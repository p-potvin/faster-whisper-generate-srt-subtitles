import unittest
import os
from unittest.mock import patch, MagicMock
from vaultwares_media_processing import core

class TestCoreIntegration(unittest.TestCase):
    @patch('vaultwares_media_processing.media.get_audio_duration_seconds')
    @patch('vaultwares_media_processing.media.srt_files_exist')
    @patch('os.path.isfile')
    @patch('vaultwares_media_processing.core.get_parakeet_model')
    @patch('vaultwares_media_processing.translation.translate_segments')
    @patch('vaultwares_media_processing.utils.write_srt')
    @patch('vaultwares_media_processing.media.fix_audio_and_reencode')
    @patch('vaultwares_media_processing.media.extract_wav_for_asr')
    @patch('os.remove')
    @patch('os.path.exists')
    @patch('time.sleep') # Mock sleep to avoid timeouts
    def test_transcribe_video_mocked(self, mock_sleep, mock_exists, mock_remove, mock_extract, mock_fix, mock_write, mock_translate, mock_get_model, mock_isfile, mock_srt_exist, mock_get_duration):
        # Setup mocks
        mock_isfile.return_value = True
        mock_srt_exist.return_value = False
        mock_get_duration.return_value = 10.0

        mock_model = MagicMock()
        mock_segment = MagicMock()
        mock_segment.start = 0.0
        mock_segment.end = 2.0
        mock_segment.text = "Hello world"
        mock_model.transcribe_file.return_value = [mock_segment]
        mock_get_model.return_value = mock_model

        # In core.py we wait for model to load:
        # while _PARAKEET_MODEL is None:
        # Instead of looping, we pre-assign _PARAKEET_MODEL
        core._PARAKEET_MODEL = mock_model

        mock_fix.return_value = "fixed_audio.wav"
        mock_extract.return_value = "asr_audio.wav"
        mock_exists.return_value = True

        # We need async mocking for translation
        async def async_translate(*args, **kwargs):
            return ["Hola mundo"]
        mock_translate.side_effect = async_translate

        input_file = "dummy_video.mp4"
        languages = ["es"]

        # Test
        output_paths = core.transcribe_video(
            input_file=input_file,
            languages=languages,
            skip_vocal_isolation=False,
            translate_api="local"
        )

        # Verify
        self.assertEqual(len(output_paths), 3) # original, english explicit, spanish
        self.assertTrue(any("dummy_video.es.srt" in p for p in output_paths))
        self.assertTrue(any("dummy_video.srt" in p for p in output_paths))

        # Verify fix_audio was called
        mock_fix.assert_called_once()
        # Verify translation was requested
        mock_translate.assert_called_once()
        # Verify files were written
        self.assertEqual(mock_write.call_count, 3)

    @patch('vaultwares_media_processing.media.get_audio_duration_seconds')
    @patch('vaultwares_media_processing.media.srt_files_exist')
    @patch('os.path.isfile')
    @patch('vaultwares_media_processing.core.get_parakeet_model')
    @patch('vaultwares_media_processing.translation.translate_segments')
    @patch('vaultwares_media_processing.utils.write_srt')
    @patch('vaultwares_media_processing.media.fix_audio_and_reencode')
    @patch('vaultwares_media_processing.media.extract_wav_for_asr')
    @patch('os.remove')
    @patch('os.path.exists')
    @patch('time.sleep')
    def test_transcribe_video_with_engine_param(self, mock_sleep, mock_exists, mock_remove, mock_extract, mock_fix, mock_write, mock_translate, mock_get_model, mock_isfile, mock_srt_exist, mock_get_duration):
        mock_isfile.return_value = True
        mock_srt_exist.return_value = False
        mock_get_duration.return_value = 5.0
        mock_fix.return_value = "fixed_audio.wav"
        mock_extract.return_value = "asr_audio.wav"
        mock_exists.return_value = True

        mock_model = MagicMock()
        mock_segment = MagicMock(start=0.0, end=1.0, text="Test segment")
        mock_model.transcribe_file.return_value = [mock_segment]
        mock_get_model.return_value = mock_model

        async def async_translate(*args, **kwargs):
            return ["Test segment"]
        mock_translate.side_effect = async_translate

        # Pass engine="parakeet" explicitly
        paths = core.transcribe_video(
            input_file="dummy.mp4",
            engine="parakeet",
            skip_vocal_isolation=True
        )
        self.assertTrue(len(paths) > 0)


if __name__ == '__main__':
    unittest.main()
