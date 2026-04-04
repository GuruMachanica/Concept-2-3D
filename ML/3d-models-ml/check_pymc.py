try:
    import pymcubes
    print('pymcubes OK', getattr(pymcubes, '__version__', '(no version)'))
except Exception as e:
    print('pymcubes import failed:', e)

try:
    import torch
    print('torch OK', 'cuda_available=' + str(torch.cuda.is_available()))
except Exception as e:
    print('torch import failed:', e)
