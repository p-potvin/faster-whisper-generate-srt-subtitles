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
    texts,
    target_lang: str,
    translate_api="deep-translator",
    max_chars=350000,
    max_calls=250,
    translate_mode="non-target",
    detector=None,
):
    if not texts:
        return []

    total_chars = sum(len(t) for t in texts)
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
            target_lang_simple = target_lang.lower().split("-")[0]
            
            # Optimization: Pre-detect languages for all non-empty segments
            non_empty_indices = [i for i, text in enumerate(texts) if text.strip()]
            non_empty_texts = [texts[i] for i in non_empty_indices]
            
            if not non_empty_texts:
                return texts

            # googletrans doesn't support bulk detection well with the Translator object in 4.0.0rc1
            # but we can at least handle the coroutine logic once and reuse the detector.
            
            translated_texts = list(texts)
            for idx in non_empty_indices:
                if calls > max_calls: 
                    raise RuntimeError("Translation request limit reached.")
                
                text = texts[idx]
                detection = detector.detect(text)
                if inspect.isawaitable(detection):
                    detection = await detection
                
                detected_lang = str(getattr(detection, "lang", "auto")).lower().split("-")[0]
                calls += 1

                if detected_lang != target_lang_simple:
                    if not is_supported_language_code(detected_lang, translate_api=translate_api):
                        # Some detections might be garbage; we could skip or fail. 
                        # Instructions say "Security First", but for translation, 
                        # skipping might be better than failing the whole file.
                        # However, to maintain current behavior:
                        raise UnsupportedLanguageError(
                            f"Detected unsupported source language '{detected_lang}' for translation."
                        )
                    
                    if calls > max_calls: 
                        raise RuntimeError("Translation request limit reached.")
                    
                    try:
                        # Jittered delay to avoid rate limiting and allow background processing
                        time.sleep(random.uniform(0.1, 0.4))
                        result = GoogleTranslator(source=detected_lang, target=target_lang).translate(text)
                        if isinstance(result, str):
                            translated_texts[idx] = result
                        else:
                            translated_texts[idx] = getattr(result, "text", text)
                        calls += 1
                    except LanguageNotSupportedException as exc:
                        raise UnsupportedLanguageError(
                            f"Detected unsupported source language '{detected_lang}' for translation."
                        ) from exc
            
            return translated_texts

        # "all" mode for deep-translator
        for text in texts:
            calls += 1
            if calls > max_calls: raise RuntimeError("Translation request limit reached.")
            if not text.strip():
                translated_texts.append(text)
                continue
            result = GoogleTranslator(source="auto", target=target_lang).translate(text)
            if isinstance(result, str):
                translated_texts.append(result)
            else:
                translated_texts.append(getattr(result, "text", text))
        return translated_texts

    else:
        raise ValueError(f"Unsupported translate API: {translate_api}")
