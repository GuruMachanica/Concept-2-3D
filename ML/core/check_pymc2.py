import sys
print('sys.path:')
for p in sys.path[:5]: print('  ', p)
for name in ('pymcubes','PyMCubes','PyMCubes','pmcubes'):
    try:
        m = __import__(name)
        print(name, 'imported as', m)
    except Exception as e:
        print(name, 'failed:', e)

import importlib
spec = importlib.util.find_spec('pymcubes')
print('find_spec(pymcubes)=', spec)
