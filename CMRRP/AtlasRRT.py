"""
Atlas RRT for Motion Planning on Constraint Manifold
Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6352929
"""

import torch

class Chart:
    """
    A single chart in an atlas

    Parameters:
    ----------
    xc: n-dimension torch.tensor
        The center point of the chart in ambient space
    em: TODO
        Exponential map ᴪ from configuration space to tangent space
    lm: TODO
        Logarithmic map ϕ from tangent space to configuration space
    """
    def __init__(self, xc, em, lm):
        self.xc = xc
        self.em = em
        self.lm = lm
        self.L = []
        self.neighbor_charts = []

class Atlas:
    """
    The atlas constrcuted for approximating the constraint manifold
    """
    def __init__(self):
        pass

class AtlasRRT:
    """
    The motion planner using Atlas-RRT method

    Parameters:
    ----------
    xs: n-dimension torch.tensor
        The starting point in ambient space
    xg: n-dimension torch.tensor
        The goal point in ambient space
    F: function
        The implicit constraint function
    Collision: function
        The collision check function 
    """
    def __init__(self, xs, xg, F, Collision):
        self.xs = xs
        self.xg = xg
        self.F = F
        self.Collision = Collision
        self.atlas = None                                       # TODO: The atlas
        self.root_1 = None                                      # TODO: root of tree 1
        self.root_2 = None                                      # TODO: root of tree 2

class AtlasRRTNode:
    """
    A tree node in Atlas-RRT

    Parameters:
    ----------
    x: n-dimension torch.tensor
        The point coordinate in ambient space
    u: k-dimension torch.tensor
        The point coordinate in tangent space
    c: Chart
        The chart that this node belongs to
    """
    def __init__(self, x, u, c):
        self.x = x
        self.u = u
        self.c = c