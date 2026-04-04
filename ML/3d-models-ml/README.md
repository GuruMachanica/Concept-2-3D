TripoSR setup
=================

This folder contains helper scripts to clone and prepare the TripoSR repository
for local use inside the `3d-models` backend. The setup replaces references to
`torchmcubes` (or similarly-named marching-cubes packages) with `pymcubes` via
a shim so the upstream code can import the expected module name.

Overview
--------
- `setup_triposr.ps1` — PowerShell script to clone TripoSR, patch its
  requirements to prefer `pymcubes`, create a virtual environment, and
  install dependencies.
- `shims/torchmcubes.py` — A small shim that proxies imports to `pymcubes`.

How to run (Windows PowerShell)
-------------------------------
1. Open PowerShell in `3d-models/backend/ml`.
2. Run:

```powershell
.\
setup_triposr.ps1
```

3. The script will clone `https://github.com/VAST-AI-Research/TripoSR.git`
   into `triposr/`, create a `.venv` at `ml/.venv`, install requirements, and
   copy the shim into the TripoSR folder so imports for `torchmcubes` resolve
   to `pymcubes`.

Notes and manual checks
-----------------------
- The script attempts a safe search-and-replace in `requirements.txt` to use
  `pymcubes` where appropriate, but you should inspect the cloned
  `triposr/requirements.txt` and the code to verify any import names.
- If TripoSR exposes a CLI entrypoint, set the backend env var `TRIPO_COMMAND`
  to a command that runs their inference CLI, using the placeholders
  `{input_image}` and `{output_dir}` which our `tripo_adapter` expects. For
  example:

```powershell
#$env:TRIPO_COMMAND = "python .\triposr\inference.py --input {input_image} --output {output_dir}"
```

Security
--------
- The setup uses `subprocess.run(shell=True)` in the adapter to allow flexible
  commands; only set `TRIPO_COMMAND` to trusted values.