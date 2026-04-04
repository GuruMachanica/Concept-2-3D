Setup a single project virtual environment (.venv)

Purpose
- Create a single `.venv` at the project root and install the combined `requirements.txt`.

Quick steps (Windows PowerShell)
```powershell
cd <repo root>
.
# from the repo root run:
PowerShell\scripts\setup_venv.ps1
# or invoke the script directly:
.\scripts\setup_venv.ps1
```

Quick steps (macOS / Linux)
```bash
cd <repo root>
./scripts/setup_venv.sh
```

Notes & recommendations
- The merged `requirements.txt` uses simple heuristics to resolve conflicting pins:
  - Exact pins (`==`) from source files were preserved when present.
  - If the same package appeared unpinned elsewhere, the pinned version was used.
  - `torch` is intentionally left unpinned because CPU vs GPU builds and CUDA versions vary widely.
 - If you need a GPU-enabled `torch` installation, install it manually inside the venv following PyTorch's official instructions.
 - After creating the venv, activate it and run any backend servers (for example, `uvicorn` from `Backend`).

If you'd like, I can now attempt to create the `.venv` and install packages automatically — this may take time and require network access. Tell me to proceed if you want that executed now.