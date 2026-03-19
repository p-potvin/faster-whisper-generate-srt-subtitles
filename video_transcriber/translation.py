import asyncio
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
            for text in texts:                
                if calls > max_calls: 
                    raise RuntimeError("Translation request limit reached.")
                if not text.strip():
                    translated_texts.append(text)
                    continue

                detection = detector.detect(text)
                if inspect.isawaitable(detection):
                    detection = await detection
                detected_lang = str(getattr(detection, "lang", "auto")).lower().split("-")[0]
                calls += 1

                if detected_lang == target_lang.lower().split("-")[0]:
                    translated_texts.append(text)
                else:
                    if not is_supported_language_code(detected_lang, translate_api=translate_api):
                        raise UnsupportedLanguageError(
                            f"Detected unsupported source language '{detected_lang}' for translation."
                        )
                    calls += 1
                    if calls > max_calls: 
                        raise RuntimeError("Translation request limit reached.")
                    try:
                        translated_texts.append(getattr(GoogleTranslator(source=detected_lang, target=target_lang).translate(text), "text", text))
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
            translated_texts.append(getattr(GoogleTranslator(source="auto", target=target_lang).translate(text), "text", text))
        return translated_texts

    else:
        raise ValueError(f"Unsupported translate API: {translate_api}")
