"""
Continuum Modular Robot Reconfigure Planner: Task and Motion Planning
"""

import sys
sys.path.append('..')
sys.path.append('../GENN')
from GMRC import GMRC
from GENN import GENN, DegreeEmbedding, SequentialPooling

class CMRRP:
    def __init__(self, g: GENN, d: DegreeEmbedding, s: SequentialPooling):
        self.genn = g
        self.degree_embedding = d
        self.sequential_pooling = s

    def plan(gmrc_1: GMRC, gmrc_2: GMRC):
        pass
