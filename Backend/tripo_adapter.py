import logging
import os
import shlex
import subprocess
import sys
import uuid
from typing import Optional


def _first_existing(paths: list[str]) -> Optional[str]:
    """Return first existing path from a list, else None."""
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _resolve_triposr_weights_dir(weights_root: str) -> Optional[str]:
    """Resolve TripoSR weights dir from either direct files or HF cache snapshot layout."""
    if not weights_root or not os.path.isdir(weights_root):
        return None

    # Direct layout: <dir>/config.yaml + <dir>/model.ckpt
    direct_cfg = os.path.join(weights_root, "config.yaml")
    direct_ckpt = os.path.join(weights_root, "model.ckpt")
    if os.path.exists(direct_cfg) and os.path.exists(direct_ckpt):
        return weights_root

    # HF cache layout:
    # <weights_root>/models--stabilityai--TripoSR/snapshots/<sha>/config.yaml
    snapshots_dir = os.path.join(weights_root, "models--stabilityai--TripoSR", "snapshots")
    if not os.path.isdir(snapshots_dir):
        return None

    candidates: list[str] = []
    for name in os.listdir(snapshots_dir):
        snap_dir = os.path.join(snapshots_dir, name)
        cfg = os.path.join(snap_dir, "config.yaml")
        ckpt = os.path.join(snap_dir, "model.ckpt")
        if os.path.isdir(snap_dir) and os.path.exists(cfg) and os.path.exists(ckpt):
            candidates.append(snap_dir)

    if not candidates:
        return None

    candidates.sort(key=os.path.getmtime, reverse=True)
    return candidates[0]


def preprocess_image(image_bytes: bytes, run_subdir: str) -> str:
    """Preprocess the uploaded image to remove background when possible.

    Returns the path to the preprocessed image (PNG) saved inside run_subdir.
    - tries `rembg` if available
    - falls back to a simple border-color thresholding using numpy/PIL
    - if all fails, writes the original bytes to `input.png` and returns that path
    """
    os.makedirs(run_subdir, exist_ok=True)
    pre_path = os.path.join(run_subdir, "input.png")
    # allow opt-out via env
    force = os.getenv("TRIPO_FORCE_REMOVE_BG", "1")
    try:
        if force.lower() in ("1", "true", "yes"):
            try:
                import rembg

                try:
                    # rembg.remove accepts bytes and returns PNG bytes
                    out_bytes = rembg.remove(image_bytes)
                    if out_bytes:
                        with open(pre_path, "wb") as pf:
                            pf.write(out_bytes)
                        return pre_path
                except Exception:
                    # try session-based API
                    try:
                        session = rembg.new_session()
                        out_bytes = rembg.remove(image_bytes, session=session)
                        if out_bytes:
                            with open(pre_path, "wb") as pf:
                                pf.write(out_bytes)
                            return pre_path
                    except Exception:
                        pass
            except Exception:
                # rembg not available or failed; fallthrough
                pass

        # Fallback simple background removal using PIL + numpy (if available)
        try:
            import io

            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes)).convert("RGBA")
            w, h = img.size
            pixels = img.load()
            # sample 4 corners to estimate background color
            corners = [
                pixels[0, 0][:3],
                pixels[w - 1, 0][:3],
                pixels[0, h - 1][:3],
                pixels[w - 1, h - 1][:3],
            ]
            avg = tuple(sum(c[i] for c in corners) // 4 for i in range(3))

            try:
                import numpy as _np

                arr = _np.array(img)
                rgb = arr[..., :3].astype(_np.int32)
                avg_arr = _np.array(avg, dtype=_np.int32)
                dif = _np.linalg.norm(rgb - avg_arr, axis=2)
                mask = dif > 30  # threshold; tune if needed
                arr[..., 3] = _np.where(mask, 255, 0)
                out_img = Image.fromarray(arr.astype(_np.uint8), mode="RGBA")
                out_img.save(pre_path)
                return pre_path
            except Exception:
                # numpy not available; fallback to very naive per-pixel loop
                for y in range(h):
                    for x in range(w):
                        r, g, b, a = pixels[x, y]
                        dif = ((r - avg[0]) ** 2 + (g - avg[1]) ** 2 + (b - avg[2]) ** 2) ** 0.5
                        if dif < 30:
                            pixels[x, y] = (r, g, b, 0)
                img.save(pre_path)
                return pre_path
        except Exception:
            # give up and write raw bytes
            pass

    except Exception:
        pass

    # Final fallback: write original image as input.png
    try:
        with open(pre_path, "wb") as pf:
            pf.write(image_bytes)
        return pre_path
    except Exception:
        raise


def generate_from_image(
    image_bytes: bytes,
    output_dir: str,
    filename_prefix: str = "tripo",
    timeout: int = 300,
) -> Optional[str]:
    """Generate a 3D model using an external TripoSR command.

    This function is intentionally generic: it expects an environment variable
    `TRIPO_COMMAND` to be set to the command that will perform generation.
    Example: `TRIPO_COMMAND=python /path/to/TripoSR/infer.py --input {input_image} --out {output_dir}`

    It writes the uploaded image to a temp file and runs the command, waiting
    up to `timeout` seconds. After completion it searches `output_dir` for a
    newly created `.glb` file and returns its filesystem path.

    Returns None if Tripo is not configured or generation failed.
    """
    tripo_cmd_template = os.getenv("TRIPO_COMMAND", "").strip()
    # allow callers to configure a timeout via env
    env_timeout = os.getenv("TRIPO_TIMEOUT")
    try:
        env_timeout = int(env_timeout) if env_timeout else None
    except Exception:
        env_timeout = None

    backend_dir = os.path.dirname(__file__)
    repo_root = os.path.dirname(backend_dir)

    # Attempt to locate an appropriate python executable for TripoSR.
    ml_python = _first_existing(
        [
            os.path.join(backend_dir, "ml", ".venv", "Scripts", "python.exe"),
            os.path.join(backend_dir, "ml", ".venv", "bin", "python"),
            os.path.join(repo_root, "ML", "core", ".venv", "Scripts", "python.exe"),
            os.path.join(repo_root, "ML", "core", ".venv", "bin", "python"),
            os.path.join(repo_root, "ML", "3d-models-ml", ".venv", "Scripts", "python.exe"),
            os.path.join(repo_root, "ML", "3d-models-ml", ".venv", "bin", "python"),
            os.path.join(repo_root, ".venv", "Scripts", "python.exe"),
            os.path.join(repo_root, ".venv", "bin", "python"),
        ]
    )

    if not tripo_cmd_template:
        # Try both old and new project layouts for TripoSR run.py
        candidate = _first_existing(
            [
                os.path.join(backend_dir, "ml", "triposr", "run.py"),
                os.path.join(repo_root, "ML", "core", "triposr", "run.py"),
                os.path.join(repo_root, "ML", "3d-models-ml", "triposr", "run.py"),
            ]
        )
        if candidate:
            python_exec = ml_python or sys.executable

            # Prefer local weights directory if present.
            weights_path = _first_existing(
                [
                    os.path.join(backend_dir, "ml", "triposr_weights"),
                    os.path.join(repo_root, "ML", "core", "triposr_weights"),
                    os.path.join(repo_root, "ML", "3d-models-ml", "triposr_weights"),
                ]
            )
            resolved_weights = _resolve_triposr_weights_dir(weights_path) if weights_path else None

            pretrained_arg = ""
            # Only pass local pretrained path when it contains the full TripoSR bundle.
            if resolved_weights:
                pretrained_arg = f' --pretrained-model-name-or-path "{resolved_weights}"'
            elif weights_path:
                logging.warning(
                    "Skipping local TripoSR weights path (no usable config.yaml/model.ckpt): %s",
                    weights_path,
                )

            # Use the ml venv python (if present) to run the script and ask for GLB output
            tripo_cmd_template = (
                f'"{python_exec}" "{candidate}" {{input_image}} '
                f"--output-dir {{output_dir}} --model-save-format glb --mc-resolution 256"
                f"{pretrained_arg}"
            )
        else:
            return None

    os.makedirs(output_dir, exist_ok=True)

    # Create a unique subdirectory under output_dir for this request to avoid collisions
    run_subdir = os.path.join(output_dir, f"{filename_prefix}_{uuid.uuid4().hex}")
    os.makedirs(run_subdir, exist_ok=True)

    # Save original upload for traceability
    orig_path = os.path.join(run_subdir, f"original_{uuid.uuid4().hex}.png")
    with open(orig_path, "wb") as f:
        f.write(image_bytes)

    # Preprocess image (remove background when possible) and use that as the input
    try:
        input_path = preprocess_image(image_bytes, run_subdir)
    except Exception:
        # fallback to original
        input_path = orig_path

    # Prepare command by replacing placeholders
    # Supported placeholders: {input_image}, {output_dir}, {python}
    python_exec = ml_python or sys.executable
    try:
        cmd = tripo_cmd_template.format(input_image=input_path, output_dir=run_subdir, python=python_exec)
    except Exception:
        cmd = tripo_cmd_template.format(input_image=input_path, output_dir=run_subdir)

    # Decide whether to run under shell or as a list (safer)
    use_shell = False
    raw_template = os.getenv("TRIPO_COMMAND", "")
    if raw_template and any(ch in raw_template for ch in ["|", ">", "&", ";", "*"]):
        use_shell = True

    # Always quote paths to avoid shell/arg parsing issues
    qinput = f'"{input_path}"'
    qout = f'"{run_subdir}"'
    try:
        try:
            cmd = tripo_cmd_template.format(input_image=qinput, output_dir=qout, python=python_exec)
        except Exception:
            cmd = tripo_cmd_template.format(input_image=qinput, output_dir=qout)

        # run and capture output
        if use_shell:
            proc = subprocess.run(
                cmd,
                shell=True,
                check=True,
                timeout=timeout or env_timeout or 300,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        else:
            try:
                cmd_list = shlex.split(cmd)
                proc = subprocess.run(
                    cmd_list,
                    shell=False,
                    check=True,
                    timeout=timeout or env_timeout or 300,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
            except Exception:
                proc = subprocess.run(
                    cmd,
                    shell=True,
                    check=True,
                    timeout=timeout or env_timeout or 300,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
        stdout = e.stdout.decode(errors="ignore") if e.stdout else ""
        logging.error("TripoSR generation failed: returncode=%s", e.returncode)
        # persist logs for debugging
        try:
            with open(os.path.join(run_subdir, "run.log"), "wb") as lf:
                lf.write(b"COMMAND:\n")
                lf.write(str(cmd).encode(errors="ignore"))
                lf.write(b"\n\n")
                lf.write(b"STDOUT:\n")
                lf.write(stdout.encode(errors="ignore"))
                lf.write(b"\nSTDERR:\n")
                lf.write(stderr.encode(errors="ignore"))
        except Exception:
            pass
        return None
    except subprocess.TimeoutExpired:
        logging.error(
            "TripoSR generation timed out after %s seconds",
            timeout or env_timeout or 300,
        )
        try:
            with open(os.path.join(run_subdir, "run.log"), "wb") as lf:
                lf.write(b"TIMEOUT\n")
                lf.write(b"COMMAND:\n")
                lf.write(str(cmd).encode(errors="ignore"))
                lf.write(b"\n")
        except Exception:
            pass
        return None
    except Exception as e:
        logging.exception("TripoSR generation error")
        try:
            with open(os.path.join(run_subdir, "run.log"), "wb") as lf:
                lf.write(str(e).encode(errors="ignore"))
        except Exception:
            pass
        return None

    # write std out/err for successful runs as well
    try:
        stdout = proc.stdout.decode(errors="ignore") if proc.stdout else ""
        stderr = proc.stderr.decode(errors="ignore") if proc.stderr else ""
        with open(os.path.join(run_subdir, "run.log"), "wb") as lf:
            lf.write(b"COMMAND:\n")
            lf.write(str(cmd).encode(errors="ignore"))
            lf.write(b"\n\n")
            lf.write(b"STDOUT:\n")
            lf.write(stdout.encode(errors="ignore"))
            lf.write(b"\nSTDERR:\n")
            lf.write(stderr.encode(errors="ignore"))
    except Exception:
        pass

    # Look for a .glb file in the run subdir (newest file)
    glb_files = []
    for root, _, files in os.walk(run_subdir):
        for f in files:
            if f.lower().endswith(".glb"):
                glb_files.append(os.path.join(root, f))
    if not glb_files:
        return None

    glb_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return glb_files[0]
