import asyncio
import time
import random
from functools import lru_cache
import inspect


class UnsupportedLanguageError(RuntimeError):
    pass


@lru_cache(maxsize=None)
def get_supported_language_codes(translate_api="deep-translator"):
    if translate_api == "deep-translator":
        from deep_translator import GoogleTranslator

        return {
            str(code).strip().lower()
            for code in GoogleTranslator().get_supported_languages(as_dict=True).values()
        }
    raise ValueError(f"Unsupported translate API: {translate_api}")


def is_supported_language_code(lang_code, translate_api="deep-translator", allow_auto=False):
    normalized_lang = str(lang_code or "").strip().lower()
    if not normalized_lang:
        return False
    if allow_auto and normalized_lang == "auto":
        return True
    return normalized_lang in get_supported_language_codes(translate_api)

async def translate_segments(
    segments,
    target_lang: str,
    translate_api="deep-translator",
    max_chars=750000,
    max_calls=1000,
    translate_mode="non-target",
    detector=None,
):
    if not segments:
        return []

    # Check if we got a list of strings or objects. Maintain backwards compatibility.
    is_list_of_strings = all(isinstance(s, str) for s in segments)
    
    total_chars = sum(len(s if is_list_of_strings else s.text) for s in segments)
    if total_chars > max_chars:
        raise RuntimeError(
            f"Translation skipped: text length {total_chars} > max_chars {max_chars}. "
            "Use --max-translate-chars to raise this limit or skip translation."
        )

    if translate_api == "deep-translator":
        try:
            from deep_translator import GoogleTranslator
            from deep_translator.exceptions import LanguageNotSupportedException
            from googletrans import Translator as GoogleTrans_Detector
        except ImportError as exc:
            raise ImportError(
                "Missing deep-translator or googletrans. Install with: "
                "pip install deep-translator googletrans==4.0.0-rc1"
            ) from exc

        translated_texts = []
        calls = 0

        if detector is None:
            detector = GoogleTrans_Detector()

        if translate_mode == "non-target":            
            # Identify which segments need translation
            indices_to_translate = []
            
            for i, s in enumerate(segments):
                text = s if is_list_of_strings else s.text
                if not text.strip():
                    continue
                
                # Logic: Skip translation if segment language matches target language
                # If we only have text, we must detect.
                if is_list_of_strings:
                    indices_to_translate.append(i)
                else:
                    # Segment language might be 'en', target might be 'en'
                    # If language is None or missing, it will be translated with source="auto"
                    current_lang = getattr(s, 'language', None) or ""
                    if current_lang.lower() != target_lang.lower():
                        indices_to_translate.append(i)

            if not indices_to_translate:
                return [s if is_list_of_strings else s.text for s in segments]

            translated_texts = [s if is_list_of_strings else s.text for s in segments]
            
            for idx in indices_to_translate:
                if calls > max_calls: 
                    raise RuntimeError("Translation request limit reached.")
                
                text = translated_texts[idx]
                
                try:                    
                    result = GoogleTranslator(target=target_lang).translate(text)

                    if isinstance(result, str):
                        translated_texts[idx] = result
                    else:
                        translated_texts[idx] = getattr(result, "text", text)

                    calls += 1
                except Exception as exc:
                    raise ValueError(
                        f"Error while translating segment {idx} with text '{text}': {exc}"
                    ) from exc
            
            print(f"Translation completed with {calls} calls.")
            
            return translated_texts

        # "all" mode for deep-translator
        from tqdm import tqdm
        input_texts = [s if is_list_of_strings else s.text for s in segments]
        for i, text in enumerate(tqdm(input_texts, desc=f"Translating to {target_lang}", unit="segment", colour="blue")):
            calls += 1
            if calls > max_calls: raise RuntimeError("Translation request limit reached.")
            if not text.strip():
                translated_texts.append(text)
                continue
            
            tqdm.write(f"Translating segment {i}: '{text}'")

            result = GoogleTranslator(source="auto", target=target_lang).translate(text)
            if isinstance(result, str):
                translated_texts.append(result)
            else:
                translated_texts.append(getattr(result, "text", text))
        return translated_texts

    else:
        raise ValueError(f"Unsupported translate API: {translate_api}")
