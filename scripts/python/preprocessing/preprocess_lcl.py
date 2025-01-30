import torch

from energydiff.dataset.lcl_electricity import PreLCL

"""
designed partition:
- 0-99: train
- 100-116: val
- 117-133: test
"""

prelcl = PreLCL(
    root = 'data/lcl/',
    list_case = list(range(0, 100)),
)
prelcl.load_process_save()

pass