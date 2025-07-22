import copy
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# Topological Modular Robot Configuration
class TMRC:
    grsp_identifier_2_id = [0, 2, 1]
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
            A list with length 3 * (w + v) for all grippers participating in grip \n
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
        self.grippers = [gripper for gripper in grippers]
        self.gripper2module = [g2m for g2m in gripper2module]
        self.module2gripper = [m2g for m2g in module2gripper]
        self.rng = rng

        if c is None:
            self.c = self.n - (self.w + self.v - 1)
        else:
            self.c = c

        if G is None:
            self.G = self.get_graph()
        else:
            self.G = copy.deepcopy(G)

        if mdl_cycles is None:
            self.mdl_cycles = self.get_mdl_cycles()
        else:
            self.mdl_cycles = [mc for mc in mdl_cycles]

        if grip_cycles is None:
            self.grip_cycles = [self.get_grip_cycle(c) for c in self.mdl_cycles]
        else:
            self.grip_cycles = [gc for gc in grip_cycles]

        if real_cycles is None:
            self.real_cycles = [self.get_real_cycle(c) for c in self.mdl_cycles]
        else:
            self.real_cycles = [rc for rc in real_cycles]

        if is_grip_w is None:
            self.is_grip_w = self.get_is_grip_w()
        else:
            self.is_grip_w = [igw for igw in is_grip_w]

    def get_is_grip_w(self):
        # Is the grip (node index) a w-grip?
        is_grip_w = [True] * (self.w + self.v)
        for i in range(len(is_grip_w)):
            if self.grippers[3 * i + 2] == -2:
                is_grip_w[i] = False
        return is_grip_w

    def get_graph(self) -> nx.MultiDiGraph:
        # Construct topology graph of MRC
        module_available = [True] * self.m
        G = nx.MultiGraph()
        for idx_g in range(len(self.grippers)):
            if self.grippers[idx_g] == -1:
                module = self.gripper2module[idx_g] // 2
                if not module_available[module]:
                    continue

                grip_1 = idx_g // 3
                grip_2 = -(2 * (self.gripper2module[idx_g] // 2) 
                           + 1 - self.gripper2module[idx_g] % 2) - 1
                
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
    
    def grip_cyc_to_mdl_cyc(self, grip_cycle):
        cycle_len = len(grip_cycle)
        mdl_cycle = [-1] * cycle_len
        mdl_cycle[0] = grip_cycle[cycle_len - 2]
        mdl_cycle[1 : cycle_len] = grip_cycle[0 : cycle_len - 1]
        return mdl_cycle

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
    
    # Execute an action on topological level
    # Return: (cyc_status, grip_status)
    # cyc_status: [cyc_sts, cyc_sts, ..., cyc_sts]: list or cyc_idx: int
    #   cyc_sts: -1: Deleted; 0: Unchanged; 1: Altered
    #   cyc_idx: Index of the new cycle
    # grip_status: (grip, status): tuple
    #   grip: The index of the grip that has been affected
    #   status: -1: v deleted; 0: w-v switched; 1: v created
    def execute_action(self, action):
        if isinstance(action, tuple):
            return TMRC._execute_grasping(self, action[0], action[1], action[2])
        else:
            return TMRC._execute_releasing(self, action)

    # gf (2 * mdl + ht) grasps gt (2 * mdl + ht)
    def _execute_grasping(self, gf, gt, gp):
        if self.module2gripper[gt % 2][gt // 2] >= 0:               # Grasp a v-grip
            self.w = self.w + 1
            self.v = self.v - 1
            self.n = self.n + 1
            emt_gripper = self.module2gripper[1 - gf % 2][gf // 2]  # Extrimity gripper
            tar_gripper = self.module2gripper[gt % 2][gt // 2] + 1  # Target gripper
            self.grippers[emt_gripper] = tar_gripper
            self.grippers[tar_gripper] = emt_gripper
            self.gripper2module[tar_gripper] = gf
            self.module2gripper[gf % 2][gf // 2] = tar_gripper
            grip_status = (tar_gripper // 3, 0)

            self.c = self.c + 1
            self.G.remove_node(-gf - 1)
            d = self.G.number_of_edges(emt_gripper // 3, tar_gripper // 3)
            if d > 0:                                               # Must be: d = 1
                edge_data = self.G.get_edge_data(emt_gripper // 3, tar_gripper // 3)
                key = 1 - next(iter(edge_data.keys()))
            else:
                key = 0
            self.G.add_edge(emt_gripper // 3, tar_gripper // 3, key=key, module=gf // 2)
            mdl_cycle = gp + [gf // 2, gp[0]]
            real_cycle = self.get_real_cycle(mdl_cycle)
            grip_cycle = self.get_grip_cycle(mdl_cycle)
            self.mdl_cycles.append(mdl_cycle)
            self.real_cycles.append(real_cycle)
            self.grip_cycles.append(grip_cycle)
            self.is_grip_w[tar_gripper // 3] = True
        else:                                                       # Grasp a leaf node
            self.v = self.v + 1
            self.n = self.n + 2
            emt_gpr_1 = self.module2gripper[1 - gf % 2][gf // 2]
            gsp_gpr_1 = 3 * (self.w + self.v - 1) + 1               # 1 Holds 2
            emt_gpr_2 = self.module2gripper[1 - gt % 2][gt // 2]
            gsp_gpr_2 = 3 * (self.w + self.v - 1)
            self.grippers.extend([-2, -2, -2])
            self.grippers[emt_gpr_1] = gsp_gpr_1
            self.grippers[gsp_gpr_1] = emt_gpr_1
            self.grippers[emt_gpr_2] = gsp_gpr_2
            self.grippers[gsp_gpr_2] = emt_gpr_2
            self.gripper2module.extend([-1, -1, -1])
            self.gripper2module[gsp_gpr_1] = gf
            self.gripper2module[gsp_gpr_2] = gt
            self.module2gripper[gf % 2][gf // 2] = gsp_gpr_1
            self.module2gripper[gt % 2][gt // 2] = gsp_gpr_2
            grip_status = (self.w + self.v - 1, 1)

            self.c = self.c + 1
            self.G.remove_node(-gf - 1)
            self.G.remove_node(-gt - 1)
            grip_1 = emt_gpr_1 // 3
            grip_2 = emt_gpr_2 // 3
            new_grip = self.w + self.v - 1
            self.G.add_edge(grip_1, new_grip, key=0, module=gf // 2)
            key = self.G.number_of_edges(grip_2, new_grip)
            self.G.add_edge(grip_2, new_grip, key=key, module=gt // 2)
            mdl_cycle = gp + [gt // 2, new_grip, gf // 2, gp[0]]
            real_cycle = self.get_real_cycle(mdl_cycle)
            grip_cycle = self.get_grip_cycle(mdl_cycle)
            self.mdl_cycles.append(mdl_cycle)
            self.real_cycles.append(real_cycle)
            self.grip_cycles.append(grip_cycle)
            self.is_grip_w.append(False)

        return (self.c - 1, grip_status)

    # Release the grasp of gb (2 * mdl + ht)
    def _execute_releasing(self, gb):
        if self.is_grip_w[self.module2gripper[gb % 2][gb // 2] // 3]:
            self.w = self.w - 1
            self.v = self.v + 1
            self.n = self.n - 1
            rls_gripper = self.module2gripper[gb % 2][gb // 2]
            emt_gripper = self.module2gripper[1 - gb % 2][gb // 2]
            self.grippers[rls_gripper] = -2
            self.grippers[emt_gripper] = -1
            self.gripper2module[rls_gripper] = -1
            self.module2gripper[gb % 2][gb // 2] = -1
            grip_status = (rls_gripper // 3, 0)

            self.c = self.c - 1
            grip_1 = rls_gripper // 3                               # 1 Holds Something
            grip_2 = emt_gripper // 3
            ge = self.gripper2module[emt_gripper]
            new_grip = -(2 * (ge // 2) + 1 - ge % 2) - 1
            edges = self.G.get_edge_data(grip_1, grip_2)
            for key, data in edges.items():
                if data.get('module') == gb // 2:
                    break
            self.G.remove_edge(grip_1, grip_2, key=key)
            self.G.add_edge(grip_2, new_grip, key=0, module=gb // 2)
            cyc_status = self._update_cycles_wo(gb // 2)
            self.is_grip_w[rls_gripper // 3] = False
        else:
            self.v = self.v - 1
            self.n = self.n - 2
            rls_gpr_1 = self.module2gripper[gb % 2][gb // 2]
            emt_gpr_1 = self.module2gripper[1 - gb % 2][gb // 2]
            module_1 = gb // 2
            rls_gpr_2 = 1 - rls_gpr_1 % 3 + 3 * (rls_gpr_1 // 3)
            emt_gpr_2 = self.grippers[rls_gpr_2]
            module_2 = self.gripper2module[rls_gpr_2] // 2
            # Grip to delete
            del_grip = rls_gpr_1 // 3
            # Leaf node to add
            grip_1 = emt_gpr_1 // 3
            ge = self.gripper2module[emt_gpr_1]
            new_leaf_1 = -(2 * (ge // 2) + 1 - ge % 2) - 1
            grip_2 = emt_gpr_2 // 3
            ge = self.gripper2module[emt_gpr_2]
            new_leaf_2 = -(2 * (ge // 2) + 1 - ge % 2) - 1
            # Update grippers
            self.grippers[emt_gpr_1] = -1
            self.grippers[emt_gpr_2] = -1
            self.grippers[3 * del_grip : 3 * del_grip + 3] = []
            for i in range(3 * (self.w + self.v)):
                if self.grippers[i] >= 3 * del_grip + 3:
                    self.grippers[i] = self.grippers[i] - 3
            # Update gripper2module
            self.gripper2module[3 * del_grip : 3 * del_grip + 3] = []
            # Update module2gripper
            for ht in range(2):
                for mdl in range(self.m):
                    if self.module2gripper[ht][mdl] == rls_gpr_1:
                        self.module2gripper[ht][mdl] = -1
                    elif self.module2gripper[ht][mdl] == rls_gpr_2:
                        self.module2gripper[ht][mdl] = -1
                    elif self.module2gripper[ht][mdl] >= 3 * del_grip + 3:
                        self.module2gripper[ht][mdl] = self.module2gripper[ht][mdl] - 3
            grip_status = (del_grip, -1)
            
            # Update c
            self.c = self.c - 1
            # Update G
            self.G.remove_node(del_grip)
            self.G.add_edge(grip_1, new_leaf_1, key=0, module=module_1)
            self.G.add_edge(grip_2, new_leaf_2, key=0, module=module_2)
            relabel_mapping = {}
            for i in range(del_grip + 1, self.w + self.v + 1):
                relabel_mapping[i] = i - 1
            nx.relabel_nodes(self.G, relabel_mapping, copy=False)
            # Update Cycles
            cyc_status = self._update_cycles_wo(None, del_grip)
            # Update is_grip_w
            self.is_grip_w[del_grip : del_grip + 1] = []

        return (cyc_status, grip_status)

    def _update_cycles_wo(self, module, grip = None):
        cyc_status = [0] * len(self.mdl_cycles)
        if grip is None:                                            # Release w-grip
            cyc_idxes = []                                          # Deleted cycles
            mdl_idxes = []                                          # Index of the mdl
            min_cyc_idx = -1                                        # Minimum del cycle
            min_mdl_idx = -1
            min_len_cyc = self.m + 1
            for i in range(len(self.mdl_cycles)):
                mdl_cycle  = self.mdl_cycles[i]
                len_cyc = len(mdl_cycle) // 2                       # Number of Modules
                for j in range(len_cyc):
                    if module == mdl_cycle[2 * j + 1]:
                        cyc_idxes.append(i)
                        mdl_idxes.append(2 * j + 1)
                        if len_cyc < min_len_cyc:
                            min_cyc_idx = i
                            min_mdl_idx = 2 * j + 1
                            min_len_cyc = len_cyc
                        break
            if len(cyc_idxes) > 1:
                min_cycle = self.mdl_cycles[min_cyc_idx]
                path_1 = min_cycle[min_mdl_idx + 1:-1] + min_cycle[:min_mdl_idx]
                path_2 = min_cycle[min_mdl_idx - 1::-1] + min_cycle[-2:min_mdl_idx:-1]
                for i in range(len(cyc_idxes)):
                    cyc_idx = cyc_idxes[i]
                    mdl_idx = mdl_idxes[i]
                    if cyc_idx != min_cyc_idx:
                        mdl_cycle = self.mdl_cycles[cyc_idx]
                        if mdl_cycle[mdl_idx - 1] == path_1[0]:
                            mdl_cycle[mdl_idx - 1 : mdl_idx + 2] = path_1
                        else:
                            mdl_cycle[mdl_idx - 1 : mdl_idx + 2] = path_2
                        mdl_cycle = self._clean_mdl_cycle(mdl_cycle)
                        self.mdl_cycles[cyc_idx] = mdl_cycle
                        self.real_cycles[cyc_idx] = self.get_real_cycle(mdl_cycle)
                        self.grip_cycles[cyc_idx] = self.get_grip_cycle(mdl_cycle)
            self.mdl_cycles[min_cyc_idx : min_cyc_idx + 1] = []
            self.real_cycles[min_cyc_idx : min_cyc_idx + 1] = []
            self.grip_cycles[min_cyc_idx : min_cyc_idx + 1] = []
        else:                                                       # Release v-grip
            # NOTE: Naturally, all grips after will be shifted forward by 1 step
            cyc_idxes = []
            grip_idxes = []
            min_cyc_idx = -1
            min_grip_idx = -1
            min_len_cyc = self.m + 1
            for i in range(len(self.grip_cycles)):
                grip_cycle = self.grip_cycles[i]
                len_cyc = len(grip_cycle) // 2
                for j in range(len_cyc):
                    if grip == grip_cycle[2 * j + 1]:
                        cyc_idxes.append(i)
                        grip_idxes.append(2 * j + 1)
                        if len_cyc < min_len_cyc:
                            min_cyc_idx = i
                            min_grip_idx = 2 * j + 1
                            min_len_cyc = len_cyc
                        break
            if len(cyc_idxes) > 1:
                min_cycle = self.grip_cycles[min_cyc_idx]
                path_1 = min_cycle[min_grip_idx + 1:-1] + min_cycle[:min_grip_idx]
                path_2 = min_cycle[min_grip_idx - 1::-1] + min_cycle[-2:min_grip_idx:-1]
                for i in range(len(cyc_idxes)):
                    cyc_idx = cyc_idxes[i]
                    grip_idx = grip_idxes[i]
                    if cyc_idx != min_cyc_idx:
                        grip_cycle = self.grip_cycles[cyc_idx]
                        ori_len = len(grip_cycle)
                        if grip_cycle[grip_idx - 1] == path_1[0]:
                            grip_cycle[grip_idx - 1 : grip_idx + 2] = path_1[2 : -2]
                        else:
                            grip_cycle[grip_idx - 1 : grip_idx + 2] = path_2[2 : -2]
                        if grip_idx == 1:
                            grip_cycle[-1] = grip_cycle[0]
                        elif grip_idx == ori_len - 2:
                            grip_cycle[0] = grip_cycle[-1]
                        mdl_cycle = self.grip_cyc_to_mdl_cyc(grip_cycle)
                        mdl_cycle = self._clean_mdl_cycle(mdl_cycle)
                        self.mdl_cycles[cyc_idx] = mdl_cycle
            self.mdl_cycles[min_cyc_idx : min_cyc_idx + 1] = []
            for i in range(len(self.mdl_cycles)):
                mdl_cycle = self.mdl_cycles[i]
                num_grips = len(mdl_cycle) // 2
                for j in range(num_grips + 1):
                    if mdl_cycle[2 * j] >= grip:
                        mdl_cycle[2 * j] = mdl_cycle[2 * j] - 1
            self.grip_cycles = [self.get_grip_cycle(c) for c in self.mdl_cycles]
            self.real_cycles = [self.get_real_cycle(c) for c in self.mdl_cycles]

        for cyc_idx in cyc_idxes:
            cyc_status[cyc_idx] = 1
        cyc_status[min_cyc_idx] = -1
        return cyc_status
    
    # Clean new module cycle after edge deletion in case there is backtracking
    def _clean_mdl_cycle(self, mdl_cycle, ban:list = None):
        if ban is None:
            ban = []
        grip_idx = [-1] * (self.w + self.v + 1) # In case is deleting a v-grip
        grip_count = [0] * (self.w + self.v + 1)
        for i in range(len(mdl_cycle) // 2):
            grip = mdl_cycle[2 * i]
            grip_count[grip] = grip_count[grip] + 1
            if grip_count[grip] > 1 and not grip in ban:
                dup_idx_1 = grip_idx[grip]
                dup_idx_2 = 2 * i               # NOTE: dup_idx_2 >= dup_idx_1
                break
            grip_idx[grip] = 2 * i
        else:
            return mdl_cycle                    # No backtracking on branch detected
        # Perform symmetricity annihilation to prevent backtracking
        clean_cycle = mdl_cycle[0 : -1]
        if not clean_cycle[dup_idx_1 - 1] == clean_cycle[dup_idx_2 + 1]:
            new_cycle = mdl_cycle[dup_idx_2 : -1] + mdl_cycle[0 : dup_idx_1 + 1]
        elif not clean_cycle[dup_idx_1 + 1] == clean_cycle[dup_idx_2 - 1]:
            new_cycle = mdl_cycle[dup_idx_1 : dup_idx_2 + 1]
        else:
            ban.append(grip)
            return self._clean_mdl_cycle(mdl_cycle, ban)
        return self._clean_mdl_cycle(new_cycle) # Could be multiple backtracking
    
    # Print w-grip modules
    def print_w_grip_modules(self):
        grip2modules = dict()
        for grip in range(self.w + self.v):
            if self.is_grip_w[grip]:
                grip2modules[grip] = [
                    int(self.gripper2module[3 * grip] // 2), 
                    int(self.gripper2module[3 * grip + 1] // 2), 
                    int(self.gripper2module[3 * grip + 2] // 2)
                ]
        print(f"Modules for w-grips are: {grip2modules}")

    def print_configuration_data(self):
        print("Module Robot Configuration Basic Data: ")
        print(f"w: {self.w}", end = '; ', flush=True)
        print(f"v: {self.v}", end = '; ', flush=True)
        print(f"n: {self.n}")
        print(f"Gripper state: {self.grippers}")

    def print_mdl_gpr_mapping(self):
        print(f"Module to grippers are: {self.module2gripper}")
        print("Grippers belong to module: ", end='')
        print([int(i // 2) for i in self.gripper2module])
        print("Grippers' head-tail conditions are: ", end='')
        print([int(i % 2) for i in self.gripper2module])

    def print_all_cycles(self):
        print(f"Module Cycles are: {self.mdl_cycles}")
        print(f"Grip Cycles are: {self.grip_cycles}")
        print(f"Real Cycles are: {self.real_cycles}")

    def show_topology(self):
        TMRC.draw_grip_graph(self.G)

    @staticmethod
    def draw_grip_graph(G):                     # Draw the topology graph of an MRC
        assert isinstance(G, nx.MultiGraph)

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
        
    @staticmethod
    def get_random_configuration_model(m, seed = None, w = None, v = None):
        rng = np.random.default_rng(seed)

        # Get the number of III-grips
        if w is not None:
            w = w
        else:
            w_max = int(np.max([2 * np.floor(m / 3), 2 * np.floor((m - 2) / 3) + 1]))
            if m == 3: w_max = 1
            p = 2 / m
            w = rng.geometric(p) - 1        # The smaller the p, the larger the E(w)
            if w > w_max: w = w_max

        # Get the number of II-grips
        if v is not None:
            v = v
        else:
            v_min = np.max([m - 2 * w - 1, 0])
            v_max = m - 2 * w + np.floor(w / 2)
            v = int(np.floor(rng.uniform(v_min, v_max + 1)))

        # Get the number of torso modules
        n = 3 * w + 2 * v - m

        # Generate a grip sequence; -2: Not Grip, -1: Unconnected, >=0: Connected Index
        # Each [x][y][z] group represents [outer gripper][middle gripper][inner gripper]
        v_grips = rng.choice(w + v, v, replace=False)
        grippers = -1 * np.ones((w + v) * 3, dtype=int)
        grippers[v_grips * 3 + 2] = -2

        # Current number of torso modules
        n_c = 0

        # Connect all grips to guarantee connectivity
        for i in range(w + v - 1):
            from_list = []
            to_list = []
            for j in range(3):
                if grippers[3 * i + j] == -1:
                    from_list.append(3 * i + j)
                if grippers[3 * (i + 1) + j] == -1:
                    to_list.append(3 * (i + 1) + j)
            from_idx = rng.choice(from_list, 1)
            to_idx = rng.choice(to_list, 1)
            grippers[from_idx] = to_idx
            grippers[to_idx] = from_idx
        n_c = n_c + (w + v - 1)
        
        # Find all left stubs, reorganize them to be ready to be drew
        # The key is that two stubs of grippers from the same grip can not be matched
        stubs = np.where(grippers == -1)[0] # All left stubs
        stub2grip = (stubs // 3).tolist()   # Corresponding grip index
        grip2stub = dict.fromkeys(stub2grip)# Dictionary from grip to a list of stubs
        grips = list(grip2stub)             # List for all grips
        for i in range(len(stubs)):
            if grip2stub[stub2grip[i]] is None:
                grip2stub[stub2grip[i]] = [stubs[i]]
            else:
                grip2stub[stub2grip[i]].append(stubs[i])
        
        # Find all grips with duplicate stubs and the ones without
        grips_w_dpcs = []                   # Grips with duplicate stubs
        grips_wo_dcps = []                  # Grips without duplicate stubs
        for grip in grips:
            if len(grip2stub[grip]) > 1:
                grips_w_dpcs.append(grip)
            else:
                grips_wo_dcps.append(grip)
        
        # Generate a random sequence for pairing stubs
        # Step 1: Arrage all stubs with brothers
        stubs2c = -1 * np.ones((len(stubs) + 1) // 2 * 2, dtype=int)
        available_pairs = list(range((len(stubs) + 1) // 2))
        for grip in grips_w_dpcs:
            pairs = rng.choice(available_pairs, 2, replace=False)
            for i in range(len(pairs)):         # From 0 to 1
                if stubs2c[2 * pairs[i]] == -1: # If both slots in the pair are empty
                    stubs2c[2 * pairs[i]] = grip2stub[grip][i]
                else:                           # If only one slot left in this pair
                    stubs2c[2 * pairs[i] + 1] = grip2stub[grip][i]
                    available_pairs.remove(pairs[i])
        # Step2: Arrange all stubs that are only children
        stubsleft = np.where(stubs2c == -1)[0]  # Find indexes for all remaining slots
        stubsleft = rng.permutation(stubsleft)
        grips_wo_dcps = rng.permutation(grips_wo_dcps)
        for i in range(len(grips_wo_dcps)):
            stubs2c[stubsleft[i]] = grip2stub[grips_wo_dcps[i]][0]

        # Connect rest torso modules
        i = 0
        while n_c < n:
            if stubs2c[i] >= 0 and stubs2c[i + 1] >= 0:     # Abandon really unlucky one
                grippers[stubs2c[i]] = stubs2c[i + 1]
                grippers[stubs2c[i + 1]] = stubs2c[i]
                n_c = n_c + 1
            i = i + 2
        
        # Allocate all grippers to the modules
        module_heads = [-1] * m         # -1: Suspend; i (i >= 0): The i-th in grippers
        module_tails = [-1] * m         # -1: Suspend; i (i >= 0): The i-th in grippers
        module2gripper = [module_heads, module_tails]
        gripper_available = [g > -2 for g in grippers]
        gripper2module = [-1] * len(grippers)   # gripper2module[idx_g] = 2 * idx_m + ht
        idx_m = 0
        for idx_g in range(len(grippers)):
            if not gripper_available[idx_g]:
                continue
            if grippers[idx_g] == -1:
                ht = rng.integers(0, 2)

                module2gripper[ht][idx_m] = idx_g
                gripper2module[idx_g] = 2 * idx_m + ht

                gripper_available[idx_g] = False
                idx_m = idx_m + 1
            elif grippers[idx_g] >= 0:
                ht = rng.integers(0, 2)

                module2gripper[ht][idx_m] = idx_g
                gripper2module[idx_g] = 2 * idx_m + ht

                module2gripper[1 - ht][idx_m] = int(grippers[idx_g])
                gripper2module[grippers[idx_g]] = 2 * idx_m + 1 - ht

                gripper_available[idx_g] = False
                gripper_available[grippers[idx_g]] = False
                idx_m = idx_m + 1
            if idx_m >= m: break

        grippers = [int(grip) for grip in grippers]

        return (w, v, n, m, grippers, gripper2module, module2gripper, rng)
    
    @staticmethod
    def get_random_configuration(m, seed = None, w = None, v = None):
        params = TMRC.get_random_configuration_model(m ,seed, w, v)
        return TMRC(*params)