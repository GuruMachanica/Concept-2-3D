"""Shim module to map `torchmcubes` imports to `pymcubes`.

This shim tries to import `pymcubes` and exposes it under the name
`torchmcubes` so upstream code that expects the other package name will
continue to work.

Placed here by the setup helper so `import torchmcubes` resolves.
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
    sys.modules.setdefault('torchmcubes', _pymcubes)
    try:
        sys.modules.setdefault('torch_mcubes', _pymcubes)
    except Exception:
        pass
