import unittest
import asyncio
import sys
from unittest.mock import MagicMock, patch

try:
    import argostranslate
except ImportError:
    argos_mock = MagicMock()
    sys.modules['argostranslate'] = argos_mock
    sys.modules['argostranslate.package'] = argos_mock.package
    sys.modules['argostranslate.translate'] = argos_mock.translate

from vaultwares_media_processing import translation

class TestTranslation(unittest.TestCase):
    def setUp(self):
        translation.get_supported_language_codes.cache_clear()

    @patch('argostranslate.package.get_available_packages')
    @patch('argostranslate.package.update_package_index')
    @patch('argostranslate.package.install_from_path')
    @patch('argostranslate.translate.get_installed_languages')
    def test_translate_segments_non_target_mode(self, mock_installed, mock_install, mock_update, mock_available):
        mock_source_lang = MagicMock()
        mock_source_lang.code = "en"

        mock_target_lang = MagicMock()
        mock_target_lang.code = "es"

        mock_translation = MagicMock()
        mock_translation.translate.return_value = "Hola Mundo"
        mock_source_lang.get_translation.return_value = mock_translation

        mock_installed.return_value = [mock_source_lang, mock_target_lang]

        texts = ["Hello world"]
        # When creating the asyncio event loop for tests
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(translation.translate_segments(
            texts, 
            target_lang="es", 
            translate_api="local",
            translate_mode="non-target"
        ))
        loop.close()

        self.assertEqual(result, ["Hola Mundo"])
        mock_translation.translate.assert_called_with("Hello world")

    @patch('argostranslate.package.get_available_packages')
    @patch('argostranslate.package.update_package_index')
    @patch('argostranslate.package.install_from_path')
    @patch('argostranslate.translate.get_installed_languages')
    def test_translate_segments_same_language_skips(self, mock_installed, mock_install, mock_update, mock_available):
        mock_source_lang = MagicMock()
        mock_source_lang.code = "en"

        mock_translation = MagicMock()
        mock_translation.translate.side_effect = lambda x: x
        mock_source_lang.get_translation.return_value = mock_translation

        mock_installed.return_value = [mock_source_lang, mock_source_lang]

        texts = ["Hola amigo"]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(translation.translate_segments(
            texts, 
            target_lang="en",
            translate_api="local",
            translate_mode="non-target"
        ))
        loop.close()

        self.assertEqual(result, ["Hola amigo"])

    @patch('argostranslate.package.get_available_packages')
    @patch('argostranslate.package.update_package_index')
    @patch('argostranslate.package.install_from_path')
    @patch('argostranslate.translate.get_installed_languages')
    @patch('tqdm.tqdm')
    def test_translate_segments_all_mode(self, mock_tqdm, mock_installed, mock_install, mock_update, mock_available):
        mock_source_lang = MagicMock()
        mock_source_lang.code = "en"

        mock_target_lang = MagicMock()
        mock_target_lang.code = "fr"

        mock_translation = MagicMock()
        mock_translation.translate.return_value = "Translated"
        mock_source_lang.get_translation.return_value = mock_translation

        mock_installed.return_value = [mock_source_lang, mock_target_lang]

        # setup tqdm correctly to bypass write error
        mock_tqdm_instance = MagicMock()
        mock_tqdm.return_value = ["Some text"]

        texts = ["Some text"]
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        result = loop.run_until_complete(translation.translate_segments(
            texts, 
            target_lang="fr", 
            translate_api="local",
            translate_mode="all"
        ))
        loop.close()

        self.assertEqual(result, ["Translated"])
        mock_translation.translate.assert_called_with("Some text")

if __name__ == '__main__':
    unittest.main()
