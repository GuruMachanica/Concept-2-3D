from pathlib import Path

from tripo_adapter import generate_from_image

out = Path("models")
out.mkdir(parents=True, exist_ok=True)

# write a tiny fake png
img = b"\x89PNG\r\n\x1a\n"
print("Running adapter test (no TRIPO_COMMAND expected)")
res = generate_from_image(img, str(out), filename_prefix="testshim", timeout=2)
print("Result:", res)
