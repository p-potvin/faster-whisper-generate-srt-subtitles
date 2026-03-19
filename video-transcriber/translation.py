import asyncio

async def translate_segments(
    texts,
    target_lang: str,
    translate_api="deep-translator",
    max_chars=350000,
    max_calls=250,
    translate_mode="all",
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
                
                detected_lang = await detector.detect(text).lang.lower().split("-")[0]
                calls += 1

                if detected_lang == target_lang.lower().split("-")[0]:
                    translated_texts.append(text)
                else:
                    calls += 1
                    if calls > max_calls: 
                        raise RuntimeError("Translation request limit reached.")
                    translated_texts.append(getattr(GoogleTranslator(source=detected_lang, target=target_lang).translate(text), "text", text))
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
