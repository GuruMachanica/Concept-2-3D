import os
from functools import lru_cache
from typing import Optional


def _extract_text(response) -> str:
    if response is None:
        return ""

    text = getattr(response, "text", None)
    if text:
        return str(text).strip()

    try:
        candidates = getattr(response, "candidates", None) or []
        for candidate in candidates:
            content = getattr(candidate, "content", None)
            parts = getattr(content, "parts", None) if content else None
            if not parts:
                continue
            joined = "".join(str(getattr(part, "text", "") or "") for part in parts).strip()
            if joined:
                return joined
    except Exception:
        pass

    return ""


@lru_cache(maxsize=4)
def _build_client(api_key: str):
    from google import genai

    return genai.Client(api_key=api_key)


def gemini_generate_text(
    prompt: str,
    model: str,
    api_key: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    image_mime_type: str = "image/png",
) -> Optional[str]:
    resolved_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not resolved_key:
        return None

    try:
        client = _build_client(resolved_key)
        if image_bytes:
            try:
                from google.genai import types as genai_types

                image_part = genai_types.Part.from_bytes(data=image_bytes, mime_type=image_mime_type)
                contents = [prompt, image_part]
            except Exception:
                contents = [prompt]
        else:
            contents = prompt

        response = client.models.generate_content(model=model, contents=contents)
        text = _extract_text(response)
        return text or None
    except Exception:
        return None


def gemini_embed_text(
    text: str,
    api_key: Optional[str] = None,
    model: str = "text-embedding-004",
) -> Optional[list[float]]:
    resolved_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
    if not resolved_key:
        return None

    try:
        client = _build_client(resolved_key)
        response = client.models.embed_content(model=model, contents=text)

        embedding = getattr(response, "embedding", None)
        if embedding is not None:
            values = getattr(embedding, "values", None)
            if values:
                return [float(v) for v in values]

        embeddings = getattr(response, "embeddings", None)
        if embeddings:
            first = embeddings[0]
            values = getattr(first, "values", None)
            if values:
                return [float(v) for v in values]
    except Exception:
        return None

    return None
