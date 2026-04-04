try:
    print('pymcubes import OK')
except Exception as e:
    print('pymcubes import failed:', e)

try:
    import torch
    print('torch import OK, cuda_available=', torch.cuda.is_available())
except Exception as e:
    print('torch import failed:', e)
