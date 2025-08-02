"""
Modular Robot Configuration Motion Planning Manifold
"""

import sys
sys.path.append('..')
import torch
from GMRC import GMRC

class MRCMPM:
    def __init__(self, gmrc_1: GMRC, gmrc_2: GMRC, action: tuple):
        self.n = gmrc_1.m
        self.avatar = gmrc_1.copy()

    def constraint_func(x: torch.tensor):
        pass
