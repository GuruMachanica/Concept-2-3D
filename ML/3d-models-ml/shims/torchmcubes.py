"""Shim module to map `torchmcubes` imports to `pymcubes`.

This shim tries to import `pymcubes` and exposes it under the name
`torchmcubes` so upstream code that expects the other package name will
continue to work.

Place the `shims` folder inside the TripoSR repo (the provided setup script
copies it there automatically).
"""
import sys
try:
    import pymcubes as _pymcubes
except Exception:
    try:
        import mcubes as _pymcubes
    except Exception:
        try:
            import PyMCubes as _pymcubes
        except Exception:
            _pymcubes = None

if _pymcubes is not None:
    # Insert into sys.modules under the expected name so `import torchmcubes`
    # will return the pymcubes/mcubes module object.
    sys.modules.setdefault('torchmcubes', _pymcubes)
    # For safety, also expose common subnames
    try:
        sys.modules.setdefault('torch_mcubes', _pymcubes)
    except Exception:
        pass
