import io

from PIL import Image

# Lazy-load the heavy `transformers` pipeline to avoid importing TensorFlow
# (or other large deps) during FastAPI/uvicorn startup. This prevents
# startup failures on systems where TF isn't installed or incompatible.
_classifier = None
_classifier_loaded = False


def _get_classifier():
    global _classifier, _classifier_loaded
    if _classifier_loaded:
        return _classifier
    _classifier_loaded = True
    try:
        from transformers import pipeline

        _classifier = pipeline("image-classification", model="google/vit-base-patch16-224")
    except Exception as e:
        # Defer logging here so startup isn't noisy; print is acceptable for local dev
        print(f"Vision model load failed: {e}")
        _classifier = None
    return _classifier


def classify_image(file_bytes: bytes) -> str:
    classifier = _get_classifier()
    if classifier is None:
        return "Unknown"

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        results = classifier(image)
        if results and len(results) > 0:
            best_label = results[0].get("label", "").split(",")[0].strip()
            return best_label or "Unknown"
    except Exception as e:
        print(f"Image classification error: {e}")

    return "Unknown"
