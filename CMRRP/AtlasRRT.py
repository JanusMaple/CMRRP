"""
Atlas RRT for Motion Planning on Constraint Manifold
Reference: https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=6352929
"""

import torch
from torch.autograd.functional import jacobian

class Chart:
    tol = 1e-6                                          # Accuracy for calculating Φ
    max_iteration = 500                                 # Time limit for exponential map
    eps = 1e-4                                          # Accuracy for exponential map

    epsilon = 1e-3                                      # Chart validity error
    alpha = torch.pi / 2                                # Chart validaty curvature
    rho = 5e-3                                          # Chart validaty spam

    rho_s = 1e-2                                        # Sample Range

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
    index: int
        A number as the identifier for this chart in the atlas
    neighbor: Chart
        The neighbor chart from where this chart is created
    """
    def __init__(self, xc: torch.tensor, J: torch.tensor, F, index, neighbor = None):
        self.xc = xc
        self.J = J
        self.F = F
        self.n = J.size()[1]
        self.k = J.size()[1] - J.size[0]
        self.device = xc.device

        self.index = index

        _, _, Vh = torch.linalg.svd(J)
        self.Phi = Vh.T[:, self.n - self.k:]                        # Orthogonal Basis

        self.L = []
        if neighbor is not None:
            self.neighbor_charts = [neighbor]

    """
    Map φ from tangent space to configuration space
    """
    def phi(self, u: torch.tensor):
        return self.xc + self.Phi @ u
    
    """
    Exponential map ᴪ maps tangent space points to the manifold
    """
    def psi(self, u: torch.tensor):
        x_init = self.phi(u)                            # Initialize using approximation
        x = x_init.clone()
        for i in range(Chart.max_iteration):
            b = torch.tensor([[self.F(x)], [self.Phi.T @ (x - self.x_init)]])
            if torch.norm(b, p=2) <= Chart.eps:
                break
            A = torch.vstack([self.J, self.Phi.T])
            x = x - A.inverse() @ b
        else:
            raise RuntimeError("Exponential Map Time Out")
        return x
    
    """
    Logarithmic map ᴪ^(-1) act as projecting point to tangent space
    """
    def psi_inv(self, x: torch.tensor):
        return self.Phi.T @ (x - self.xc)
    
    """
    Whether a point is in the validity area of the chart
    """
    def in_V(self, u: torch.tensor):
        if torch.norm(u, 2) >= Chart.rho:
            return False
        
        x = self.psi(u)
        if torch.norm(x - self.phi(u), 2) >= Chart.epsilon:
            return False
        
        Jx = jacobian(self.F, x)
        _, _, Vh = torch.linalg.svd(Jx)
        Phi_x = Vh.T[:, self.n - self.k:]
        if torch.linalg.matrix_norm(self.Phi.T @ Phi_x, ord=2) > torch.cos(Chart.alpha):
            return False

        return True
    
    """
    Whether a point is in the polyedral of the chart
    """
    def in_P(self, u: torch.tensor):
        for l in self.L:
            pass        # TODO

        return True

    """
    Sample a point in the chart tangent space within the ball
    """
    def sample(self):
        z = torch.randn(self.k, device=self.device, requires_grad=True)
        r = Chart.rho_s * torch.rand(1, device=self.device).pow(1.0 / self.k)
        u = r * z / z.norm()
        return u

class Atlas:
    """
    The atlas constrcuted for approximating the constraint manifold

    Parameters:
    ----------
    xs: n-dimension torch.tensor
        The starting point in ambient space
    xg: n-dimension torch.tensor
        The goal point in ambient space
    F: function
        The implicit constraint function
    """
    def __init__(self, xs, xg, F):
        self.charts = [Chart(xs, jacobian(F, xs), F, 0), 
                       Chart(xg, jacobian(F, xg), F, 1)]
        self.F = F

    def add_chart(self, x):
        pass

    def get_chart(self, index) -> Chart:
        return self.charts[index]

class AtlasRRTNode:
    """
    A tree node in Atlas-RRT

    Parameters:
    ----------
    node_index: int
        The node's index in its tree's node list
    x: n-dimension torch.tensor
        The point coordinate in ambient space
    c: Chart
        The chart that this node belongs to
    u: k-dimension torch.tensor
        The point coordinate in tangent space
    parent: AtlasRRTNode
        The parent node of this node in tree
    """
    def __init__(self, node_index, x: torch.tensor, c: Chart, u = None, parent = None):
        self.node_index = node_index                # The node index in tree node list
        self.x = x
        self.chart = c
        if u is None:
            self.u = c.psi_inv(x)
        else:
            self.u = u
        self.parent = parent
        self.children = []

class AtlasRRTree:
    delta = 1e-3                                    # TODO
    lambda_ = 3.0                                   # TODO

    """
    Tree in Atlas-RRT

    Parameters:
    ----------
    x_root: n-dimension torch.tensor
        The root point coordinate in ambient space
    atlas: Atlas
        The atlas that this tree is referring to
    chart: Chart
        The chart where the root node belongs to
    Collision: function
        The collision check function 
    """
    def __init__(self, x_root: torch.tensor, atlas: Atlas, chart: Chart, Collision):
        self.root = AtlasRRTNode(0, x_root, chart)
        self.atlas = atlas                          # The binded atlas for this tree
        self.Collision = Collision                  # Collision check function
        self.nodes = [self.root]                    # All nodes in this tree
        self.chart_indexes = [chart.index]          # In which chart each node is
        self.node_num = 1                           # Total number of nodes in tree
        self.device = chart.device

    """
    Sample a point in the atlas within all charts that have been reached by this tree
    """
    def sample(self):
        node_charts = torch.tensor(self.chart_indexes, 
                                   dtype=torch.long, device=self.device)
        counts = torch.bincount(node_charts)
        nonzero_mask = counts > 0
        weights = torch.zeros_like(counts, device=self.device)
        weights[nonzero_mask] = 1.0 / (counts[nonzero_mask].float() + 1.0)
        probs = weights / weights.sum()
        while True:
            r = torch.multinomial(probs, 1).item()
            chart = self.atlas.get_chart(r)
            ur = chart.sample()
            if chart.in_P(ur):
                break
        return chart, ur
    
    """
    Get the nearest node to a specific point (from neighboring charts if there is any)

    Parameters:
    ----------
    xr: n-dimension torch.tensor
        The random point thet to be extended
    chart: Chart
        The chart where this random point is sampled
    """
    def get_nearest_node(self, xr, chart: Chart = None) -> int:
        nearest_dis = torch.inf
        for i in range(len(self.nodes)):
            if chart is not None:                   # Search only ego and neighbor charts
                node_chart_index = self.chart_indexes[i]
                is_skip = True
                if chart.index == node_chart_index:
                    is_skip = False
                for neighbor_chart in chart.neighbor_charts:
                    if neighbor_chart.index == node_chart_index:
                        is_skip = False
                if is_skip:
                    continue
            node = self.nodes[i]
            dis = torch.norm(node.x - xr)           # Approximate using metric distance
            if dis < nearest_dis:
                nearest_dis = dis
                nearest_node_index = i
        return nearest_node_index
    
    """
    Extend from current tree to a given ambient space point and return ending node index

    Parameters:
    ----------
    xr: n-dimension torch.tensor
        The random point thet to be extended
    node_index: int
        The index of the node from where to be extended
    is_explore: bool
        Whether the extension is exploring or connecting
    """
    def extend(self, xr, node_index, is_explore) -> int:
        pass

    """
    Get the path from root to a node, its index is given as the parameter
    """
    def get_path(self, node_index) -> list:
        return []                                   # TODO

class AtlasRRTPlanner:
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
    def __init__(self, xs: torch.tensor, xg: torch.tensor, F, Collision):
        self.xs = xs
        self.xg = xg
        self.atlas = Atlas(xs, xg, F)
        self.Ts = AtlasRRTree(xs, self.atlas, self.atlas.get_chart(0), Collision)
        self.Tg = AtlasRRTree(xg, self.atlas, self.atlas.get_chart(1), Collision)

    def plan(self):
        trees = [self.Ts, self.Tg]
        i = 0
        while True:
            chart, ur = trees[i].sample()
            xr = chart.psi(ur)
            node_index_0 = trees[i].get_nearest_node(xr, chart)
            ni0 = trees[i].extend(xr, node_index_0, is_explore=True)
            xl0 = trees[i].nodes[ni0].x
            node_index_1 = trees[1 - i].get_nearest_node(xr)
            ni1 = trees[1 - i].extend(xr, node_index_1, is_explore=False)
            xl1 = trees[1 - i].nodes[ni1].x
            if torch.norm(xl0 - xl1, 2) < AtlasRRTree.delta:
                break
            i = 1 - i
        return trees[i].get_path(ni0) + trees[1 - i].get_path(ni1).reverse()
    