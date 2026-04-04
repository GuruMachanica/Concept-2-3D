import sys
import site
print('sys.executable=', sys.executable)
print('site.getsitepackages()=')
try:
    for p in site.getsitepackages(): print('  ', p)
except Exception as e:
    print('site.getsitepackages failed:', e)
print('sys.path:')
for p in sys.path[:10]: print('  ', p)
