"""
Atlas RRT for Motion Planning on Constraint Manifold
Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6352929
"""

import torch

class Chart:
    tol = 1e-6                                          # Accuracy for calculating Φ
    max_iteration = 500                                 # Time limit for exponential map
    epsilon = 1e-4                                      # Accuracy for exponential map

    """
    A single chart in an atlas

    Parameters:
    ----------
    xc: n-dimension torch.tensor
        The center point of the chart in ambient space
    J: (n-k, n)-dimension torch.tensor
        The Jacobian of constraint function at point xc
    F: function
        The implicit constraint function
    """
    def __init__(self, xc, J, F):
        self.xc = xc
        self.J = J
        self.F = F

        _, S, Vh = torch.linalg.svd(J)
        rank = (S > Chart.tol).sum().item()
        self.Phi = Vh.T[:, rank:]                       # Orthogonal Basis for the chart

        self.L = []
        self.neighbor_charts = []

    """
    Map φ from tangent space to configuration space
    """
    def phi(self, u):
        return self.xc + self.Phi @ u
    
    """
    Exponential map ᴪ maps tangent space points to the manifold
    """
    def psi(self, u):
        x_init = self.phi(u)                            # Initialize using approximation
        x = x_init.clone()
        for i in range(Chart.max_iteration):
            b = torch.tensor([[self.F(x)], [self.Phi.T @ (x - self.x_init)]])
            if torch.norm(b, p=2) <= Chart.epsilon:
                break
            A = torch.vstack([self.J, self.Phi.T])
            x = x - A.inverse() @ b
        else:
            raise RuntimeError("Exponential Map Time Out")
        return x
    
    """
    Logarithmic map ᴪ^(-1) act as projecting point to tangent space
    """
    def psi_inv(self, x):
        return self.Phi.T @ (x - self.xc)

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