import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Topological Modular Robot Configuration
class TMRC:
    def __init__(self, w, v, n, m, 
                 grippers, gripper2module, module2gripper, rng: np.random.Generator, 
                 c = None, G = None, mdl_cycles = None, grip_cycles = None
                 , real_cycles = None, is_grip_w = None):
        """
        The topology of a configuration. Cycle basis will be found. 

        Parameters
        ----------
        w: int
            Number of 3-grips in MRC \n

        v: int
            Number of 2-grips in MRC \n

        n: int
            Number of in-grip modules in MRC \n

        m: int
            Number of modules in MRC \n

        grippers : [int, int, ..., int]; size: 3 * (w + v)
            Actually, a more appropriate name should be gripper2gripper \n
            A list with length  for all grippers participating in grip \n
            grippers[i] = -2: No gripper here at i \n
            gripper[i] = -1: Gripper i is connected to a suspending gripper \n
            gripper[i] >= 0: Gripper i is connected to gripper gripper[i] \n
            Indexes [g0, g1, g2]: g0 held by g1, g1 held by g2 \n
            Indexes [g0, g1, -2]: g0 held by g1 \n

        gripper2module : [int, int, ..., int]; size: 3 * (w + v)
            gripper2module[i] = g2m: Gripper i is from module (g2m // 2) \n
            g2m % 2 == 0: Gripper i is the head gripper \n
            g2m % 2 == 1: Gripper i is the tail gripper \n

        module2gripper : [[int, int, ... int], [int, int, ..., int]]; size: (2, m)
            module2gripper[0][i] >= 0: The head gripper of module i \n
            module2gripper[1][i] >= 0: The tail gripper of module i \n
            module2gripper[ht][i] < 0: The gripper does not participate in grips \n

        rng: numpy.random.Generator
            A shared generator for MRC and TMRC \n
            
        Auto-Generated Properties
        -------
        c: int
            Circuit rank of the graph, which will always be: c = n - (w + v - 1) \n

        G: networkx.Multigraph
            A multigraph for the modular robot configuration \n

        mdl_cycles: [cyc_list, cyc_list, ..., cyc_list]; size: (1, circuit_rank)
            cyc_list: [*grip, module, grip, module, ..., module. *grip] \n
            or: [*node, edge, node, edge, ..., edge, node] \n

        grip_cycles: [cyc_list, cyc_list, ..., cyc_list]; size: (1, circuit_rank)
            cyc_list: [*module, grip, module, grip, ..., grip, module] \n
            grip_cycles is basically mdl_cycles shifted by 1

        real_cycles: [cyc_list, cyc_list, ..., cyc_list]; size: (1, circuit_rank)
            cyc_list: [int, int, ..., int], the size is (3 * m_c) \n
            m_c: The number of modules in the cycle \n
            It is [gripper, module, gripper] repeated for m_c times \n
        
        is_grip_w: [bool, bool, ..., bool]; size: (w + v)
            Whether a grip is a w-grip or not \n
        """
        self.w = w
        self.v = v
        self.n = n
        self.m = m
        self.grippers = grippers
        self.gripper2module = gripper2module
        self.module2gripper = module2gripper
        self.rng = rng

        if c is None:
            self.c = self.n - (self.w + self.v - 1)
        else:
            self.c = c

        if G is None:
            self.G = self.get_graph()
        else:
            self.G = G

        if mdl_cycles is None:
            self.mdl_cycles = self.get_mdl_cycles()
        else:
            self.mdl_cycles = mdl_cycles

        if grip_cycles is None:
            self.grip_cycles = [self.get_grip_cycle(c) for c in self.mdl_cycles]
        else:
            self.grip_cycles = grip_cycles

        if real_cycles is None:
            self.real_cycles = [self.get_real_cycle(c) for c in self.mdl_cycles]
        else:
            self.real_cycles = real_cycles

        if is_grip_w is None:
            self.is_grip_w = self.get_is_grip_w()
        else:
            self.is_grip_w = is_grip_w

    def get_is_grip_w(self):
        # Is the grip (node index) a w-grip?
        is_grip_w = [True] * (self.w + self.v)
        for i in range(len(is_grip_w)):
            if self.grippers[3 * i + 2] == -2:
                is_grip_w[i] = False
        return is_grip_w

    def get_graph(self):
        # Construct topology graph of MRC
        module_available = [True] * self.m
        G = nx.MultiGraph()
        for idx_g in range(len(self.grippers)):
            if self.grippers[idx_g] == -1:
                module = self.gripper2module[idx_g] // 2
                if not module_available[module]:
                    continue

                grip_1 = idx_g // 3
                grip_2 = -idx_g - 1
                
                G.add_edge(grip_1, grip_2, key=0, module=int(module))

                module_available[module] = False
            elif self.grippers[idx_g] >= 0:
                module = self.gripper2module[idx_g] // 2
                if not module_available[module]:
                    continue

                grip_1 = idx_g // 3
                grip_2 = int(self.grippers[idx_g]) // 3

                key = G.number_of_edges(grip_1, grip_2)
                G.add_edge(grip_1, grip_2, key=key, module=int(module))

                module_available[module] = False
        return G

    def get_mdl_cycles(self):
        # Find minimum cycle basis for the graph G
        spl_cycles = list(nx.simple_cycles(self.G))     # Parallel edges not considered
        rnd_idx = self.rng.permutation(len(spl_cycles))
        spl_cycles = [spl_cycles[i] for i in rnd_idx]   # Break ties for the sort after
        spl_cyc_lens = [0] * len(spl_cycles)
        for i in range(len(spl_cycles)):
            spl_cyc_lens[i] = len(spl_cycles[i])
        sort_idx = np.argsort(spl_cyc_lens)
        spl_cycles = [spl_cycles[i] for i in sort_idx]  # Sort by cycle length
        # Each cycle follows: [grip*, module, grip, module, ..., grip*]
        mdl_cycles = []                                 # Parallel edges are considered
        # NOTE: All nodes in cycle has a number in [0, number of grips)
        illegal_hashes=[]                               # 2-power as hash for each node
        spanned_sets = []                               # All sets of edges of a cycle
        for spl_cyc in spl_cycles:
            # Add a new simple cycle into mdl_cycles to form a new constraint
            if len(spl_cyc) == 2:
                grip_1 = spl_cyc[0]
                grip_2 = spl_cyc[1]
                module_0 = self.G[grip_1][grip_2][0]['module']
                module_1 = self.G[grip_1][grip_2][1]['module']
                mdl_cycle = [grip_1, module_0, grip_2, module_1, grip_1]
                mdl_cycles.append(mdl_cycle)
            else:
                # Rule out simple cycles that can already be spanned by cycle basis
                spl_cyc_hash = 0
                for node_idx in spl_cyc:
                    spl_cyc_hash = spl_cyc_hash + 2 ** node_idx
                if spl_cyc_hash in illegal_hashes:
                    continue
                else:
                    illegal_hashes.append(spl_cyc_hash)
                # Count the number of parallel edges in the cycle
                num_pes = 0
                for i in range(len(spl_cyc)):
                    grip_1 = spl_cyc[i]
                    grip_2 = spl_cyc[(i + 1) % len(spl_cyc)]
                    if self.G.number_of_edges(grip_1, grip_2) > 1:
                        num_pes = num_pes + 1
                # Choose one of the edge for all pairs of parallel edges
                module_choice = self.rng.choice(2 ** num_pes)
                mdl_cycle = []
                mdl_cycle.append(spl_cyc[0])
                for i in range(len(spl_cyc)):
                    grip_1 = spl_cyc[i]
                    grip_2 = spl_cyc[(i + 1) % len(spl_cyc)]
                    if self.G.number_of_edges(grip_1, grip_2) > 1:
                        module = self.G[grip_1][grip_2][module_choice % 2]['module']
                        module_choice = module_choice // 2
                    else:
                        module = self.G[grip_1][grip_2][0]['module']
                    mdl_cycle.append(module)
                    mdl_cycle.append(grip_2)
                mdl_cycles.append(mdl_cycle)
                # Update all sets that can be spanned by current cycle basis
                spl_cyc_set = set(
                    [(min(spl_cyc[i], spl_cyc[(i + 1) % len(spl_cyc)]),
                      max((spl_cyc[i], spl_cyc[(i + 1) % len(spl_cyc)]))) 
                    for i in range(len(spl_cyc))])
                adhered_sets = []
                for ss in spanned_sets:
                    if not spl_cyc_set.isdisjoint(ss):
                        adhered_sets.append(ss)
                for ads_choice in range(2 ** (len(adhered_sets)) - 1):
                    new_set = spl_cyc_set
                    adc = ads_choice
                    for i in range(len(adhered_sets)):
                        if adc % 2 == 0:
                            new_set = new_set.symmetric_difference(adhered_sets[i])
                        adc = adc // 2
                    new_set_nodes = set()
                    for new_set_edge in new_set:
                        new_set_nodes.add(new_set_edge[0])
                        new_set_nodes.add(new_set_edge[1])
                    ns_hash = 0
                    for node_idx in new_set_nodes:
                        ns_hash = ns_hash + 2 ** node_idx
                    # It is possible to have same new_sets with different ads_choice
                    if ns_hash not in illegal_hashes:
                        illegal_hashes.append(ns_hash)
                        spanned_sets.append(new_set)
                spanned_sets.append(spl_cyc_set)
            if len(mdl_cycles) >= self.c:
                break
        return mdl_cycles

    def get_real_cycle(self, mdl_cycle):
        m_c = len(mdl_cycle) // 2
        real_cycle = [-1] * (3 * m_c)
        for i in range(m_c):
            module = mdl_cycle[2 * i + 1]
            gripper_1 = self.get_gripper(module, mdl_cycle[2 * i])
            gripper_2 = self.get_gripper(module, mdl_cycle[2 * i + 2])
            real_cycle[3 * i] = gripper_1
            real_cycle[3 * i + 1] = module
            real_cycle[3 * i + 2] = gripper_2
        return real_cycle
    
    def get_grip_cycle(self, mdl_cycle):
        cycle_len = len(mdl_cycle)
        grip_cycle = [-1] * cycle_len
        grip_cycle[0] = mdl_cycle[cycle_len - 2]
        grip_cycle[1 : cycle_len] = mdl_cycle[0 : cycle_len - 1]
        return grip_cycle
    
    # Find the module's gripper in the grip
    def get_gripper(self, module, grip):
        if self.module2gripper[0][module] // 3 == grip:
            return self.module2gripper[0][module]
        else:
            return self.module2gripper[1][module]
        
    # Get grip neighbors of a grip and also the connecting modules between them
    # Return: [[mdl, grip], ..., [mdl, grip]], meaning it is connected by mdl to grip
    def get_grip_neighbors_w_mdl(self, grip):
        nwm = []
        for g in range(3 * grip, 3 * grip + 3):
            gn = self.grippers[g]
            if gn >= 0:
                nwm.append([int(self.gripper2module[g] // 2), int(gn // 3)])
        return nwm

    # Get direct neibor grippers for a gripper within a grip:
    def get_gripper_grip_neighbors(self, g):
        if self.grippers[g] == -2:
            return []
        grip = g // 3
        if self.is_grip_w[grip]:
            ggns = []
            for i in range(3 * grip, 3 * grip + 3):
                if not i == g and np.abs(i - g) < 2:            # With direct contact
                    ggns.append(i)
            return ggns
        else:
            return [grip * 3 + 1 - g % 3]
    
    # Get all neighbor grippers for a gripper, including its module neighbor in grips
    # NOTE: Limb grippers (which does not participate in any grips are not considered)
    def get_gripper_neighbors(self, g):
        gns = self.get_gripper_grip_neighbors(g)
        if self.grippers[g] >= 0:       # Only grippers in grips are considered
            gns.append(int(self.grippers[g]))
        return gns
    
    @staticmethod
    def show_topology(G):                   # Draw the topology graph of an MRC
        assert isinstance(G, nx.MultiGraph)
        
        print("Edges in the graph: ", end='')
        print(G.edges)

        pos = nx.spring_layout(G)
        fig, ax = plt.subplots()
        nx.draw_networkx_nodes(G, pos, ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        curved_edges = [edge for edge in G.edges() if G.number_of_edges(*edge) > 1]
        straight_edges = list(set(G.edges()) - set(curved_edges))
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=straight_edges)
        arc_rad = 0.25
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(set(curved_edges)),
                            connectionstyle=f'arc3, rad = {arc_rad}')
        nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(set(curved_edges)),
                            connectionstyle=f'arc3, rad = {-arc_rad}')
        
        edge_labels = {}
        for u, v, key, data in G.edges(keys=True, data=True):
            if G.number_of_edges(u, v) == 1:
                label = f"{data.get('module')}"
            else:
                label = f"{G[u][v][0]['module']}, {G[u][v][1]['module']}"
            edge_labels[(u, v, key)] = label

        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=edge_labels, label_pos=0.5, font_size=8)