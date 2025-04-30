from TMRC import TMRC

# Embedded Modular Robot Configuration
# The term "Polarity" is abused for representing:
#   1. The order of the edges in a cycle list
#   2. The rotation system of the grips
#   3. The orientation of modules in the cycles

class EMRC(TMRC):
    """
    Combinatorial Embedding of the Configuration on a Orientable 2D Surface.  
    
    Parameters
    ----------
    module_polarities: [int, int, ..., int]; size is m
        0: not in loop \n
        1: h-t in 360 loop or t-h in -360 loop \n
        -1: otherwise \n

    loop_polarities: [int, int, ..., int]; size is c
        0: to be decided \n
        1: 360 loop (counterclockwise) \n
        -1: -360 loop (clockwise) \n

    grip_polarities: [int, int, ..., int]; size is w + v
        0: not in loop or is a v-grip \n
        1: gripper_1 -> gripper_2 -> gripper_3 in counterclockwise direction \n
            such that gamma_1 + gamma_2 + 180 ∈ [-115, 115] degrees \n
        -1: # 1: gripper_1 -> gripper_2 -> gripper_3 in clockwise direction \n
            such that gamma_1 + gamma_2 + 180 ∈ [245, 475] degrees \n
        Reason: gamma_1 + 180 + gamma_2 + 180 + gamma_3 + 180 = 360 (cc) or 720 (c) \n
    """
    def __init__(self, w, v, n, m, grippers, gripper2module, module2gripper, rng,
                 module_polarities = None, 
                 loop_polarities = None, 
                 grip_polarities = None):
        super().__init__(w, v, n, m, grippers, gripper2module, module2gripper, rng)
        
        self.module_loops = [self.get_module_loop(c) for c in self.real_cycles]
        self.module_ht_loops = [self.get_module_ht_loop(c) for c in self.real_cycles]

        self.grasp_loops = [self.get_grasp_loop(c) for c in self.real_cycles]
        self.grasp_dir_loops = [self.get_grasp_dir_loop(l) for l in self.grasp_loops]

        if (module_polarities is None 
            or loop_polarities is None 
            or grip_polarities is None):
            self.module_polarities, self.loop_polarities, self.grip_polarities = \
                self.get_polarities()
        else:
            self.module_polarities = module_polarities
            self.loop_polarities = loop_polarities
            self.grip_polarities = grip_polarities

        self.actions = self.get_all_actions()

    # [m0, m1, ..., mc], if mi > 0 than it is from head to tail; otherwise tail to head
    def get_module_loop(self, real_cycle):
        module_loop = [0] * (len(real_cycle) // 3)
        for i in range(len(module_loop)):
            module_loop[i] = real_cycle[3 * i + 1]
        return module_loop
    
    # Get module direction in a loop; 1: head to tail; -1: tail to head
    def get_module_ht_loop(self, real_cycle):
        module_ht_loop = [0] * (len(real_cycle) // 3)
        for i in range(len(module_ht_loop)):
            if self.gripper2module[real_cycle[3 * i]] % 2 == 0:     # If head to tail
                module_ht_loop[i] = 1
            else:
                module_ht_loop[i] = -1
        return module_ht_loop

    # [(g1, g2), (g3, g4), ...], (g1, g2) is from gripper g1 to gripper g2
    def get_grasp_loop(self, real_cycle):
        grasp_loop = []
        for i in range(len(real_cycle) // 3):
            if i == 0:
                gripper_1 = real_cycle[len(real_cycle) - 1]
            else:
                gripper_1 = real_cycle[3 * i - 1]
            gripper_2 = real_cycle[3 * i]
            grasp_loop.append((gripper_1, gripper_2))
        return grasp_loop
    
    # Get grasp direction in a loop; 1: outer to inner; -1: inner to outer
    def get_grasp_dir_loop(self, grasp_loop):
        grasp_dir_loop = [0] * len(grasp_loop)
        for i in range(len(grasp_loop)):
            # If 0 -> 1 or 1 -> 2 or 2 -> 0
            if grasp_loop[i][0] % 3 == (grasp_loop[i][1] - 1) % 3:
                grasp_dir_loop[i] = 1
            # If 0 -> 2 or 1 -> 0 or 2 -> 1
            elif grasp_loop[i][0] % 3 == (grasp_loop[i][1] + 1) % 3:
                grasp_dir_loop[i] = -1
        return grasp_dir_loop
    
    # Try to make the minimum cycle basis exactly face cycle basis
    # Also try to put non-cyclic edges on the outer face of the embedding
    def get_polarities(self):
        module_polarities = [0] * self.m
        mdl_plr_weights = [0] * self.m
        loop_polarities = [0] * self.c
        grip_polarities = [0] * (self.w + self.v)
        # Large cycles first
        for i in reversed(range(len(self.module_loops))):
            # Decide Loop Polarity
            module_loop = self.module_loops[i]
            vote = 0
            for j in range(len(module_loop)):
                module = module_loop[j]
                # If the module already h-t in 360 or t-h in -360
                if module_polarities[module] == 1:
                    # Loop should be 360 for t-h or -360 for h-t
                    vote = vote - self.module_ht_loops[i][j]
                elif module_polarities[module] == -1:
                    vote = vote + self.module_ht_loops[i][j]
            if vote == 0:
                loop_polarities[i] = int(self.rng.choice([-1, 1], 1)[0])    # Random
            elif vote < 0:
                loop_polarities[i] = -1
            else:
                loop_polarities[i] = 1
            
            # Decide Module Polarity
            for j in range(len(module_loop)):   # Update module polarities
                module = module_loop[j]
                new_module_polarity = loop_polarities[i] * self.module_ht_loops[i][j]
                if module_polarities[module] == 0:
                    module_polarities[module] = new_module_polarity
                    mdl_plr_weights[module] = len(module_loop)
                elif not module_polarities[module] == new_module_polarity:
                    w_old = mdl_plr_weights[module]
                    w_new = len(module_loop)
                    # Smaller Loop has Larger Chance to Dominate the Fate of a Module
                    if self.rng.random() > w_new / (w_old + w_new):
                        module_polarities[module] = new_module_polarity
                        mdl_plr_weights[module] = w_new
                else:
                    mdl_plr_weights[module] = len(module_loop)      # Enhance this Fate

            # Decide Grip Polarity
            for j in range(len(module_loop)):
                grip = self.grip_cycles[i][2 * j + 1]
                if self.is_grip_w[grip]:
                    if grip_polarities[grip] == 0:
                        module_1 = self.grip_cycles[i][2 * j]
                        gripper_1 = self.get_gripper(module_1, grip) % 3
                        module_2 = self.grip_cycles[i][2 * j + 2]
                        gripper_2 = self.get_gripper(module_2, grip) % 3
                        if gripper_2 == (gripper_1 + 1) % 3:
                            grip_polarities[grip] = -loop_polarities[i]
                        else:
                            grip_polarities[grip] = loop_polarities[i]
        for grip in range(self.w + self.v):
            if self.is_grip_w[grip]:
                if grip_polarities[grip] == 0:
                    grip_polarities[grip] = int(self.rng.choice([-1, 1], 1)[0])

        return module_polarities, loop_polarities, grip_polarities
    
    # Get the polarity vote of a w-grip if goes from m1 to m2
    def get_polarity_vote(self, m1, grip, m2):
        g1 = self.get_gripper(m1, grip)
        g2 = self.get_gripper(m2, grip)
        if g2 % 3 == (g1 + 1) % 3:
            return -self.grip_polarities[grip]
        else:
            return self.grip_polarities[grip]
    
    # Return: [(gf, gt, gp, pp, pv), ..., (gf, gt, gp, pp, pv), gb, ..., gb]
    # gf: 2 * module + ht, the docking starts from this gripper
    # gt: 2 * module + ht, the docking goes to this gripper, which is be grasped by gf
    # gp: grip_path [grip, module, grip, ..., module, grip]
    #   First loop is +1 polarity and secone is -1 polarity
    #   The starting grip is the only neighbor of the leaf node gf
    #   The end grip is non-leaf node gt or the only neighbor of the leaf node gt
    # pp: int; the path polarity, either -1 (clockwise) or +1 (conterclockwise)
    # pv: (int, int), the path polarity and its vote counts
    #   (-1 votes, +1 votes)
    # gb: 2 * module + ht, gripper to be broken
    def get_all_actions(self):
        actions = []
        
        # Find all branch ends and v-grips for a new docking action
        # NOTE: It does not matter that who is inside, it only matters who is out-out
        gfs = []            # All leaf node grippers
        gvs = []            # All v-grip outside grippers
        for i in range(self.m):
            for j in range(2):
                if self.module2gripper[j][i] < 0:
                    gfs.append(2 * i + j)
        for i in range(self.w + self.v):
            if not self.is_grip_w[i]:               # All v-grips can be catched
                gvs.append(int(self.gripper2module[3 * i + 1]))
        # BFS for finding grip_paths as the shortest path connecting gf and gt
        for i in range(len(gfs)):
            gf = gfs[i]
            grip_start = self.module2gripper[1 - gf % 2][gf // 2] // 3

            gts = []
            gts.extend(gfs[i + 1:])                 # Truncated by half due to symmetry
            # It is possible for duplicate elements in grip_ends
            #   Case 1: A leaf node connected by a v-grip
            #   Case 2: Two leaf nodes sharing the same w-grip as the base
            grip_ends = [self.module2gripper[1 - gt % 2][gt // 2] // 3 for gt in gts]
            for gv in gvs:
                grip_end = self.module2gripper[gv % 2][gv // 2] // 3
                if grip_start == grip_end:
                    continue
                gts.append(gv)
                grip_ends.append(grip_end)

            gps = [None] * len(gts)
            grip_path = [None] * (self.w + self.v)
            grip_visited = [False] * (self.w + self.v)
            grip_needs_visiting = [False] * (self.w + self.v)
            num_gnv = 0             # Number of grippers that need visiting
            for grip_end in grip_ends:
                if not grip_needs_visiting[grip_end]:
                    grip_needs_visiting[grip_end] = True
                    num_gnv = num_gnv + 1
            # front: an int list [grip, grip, ..., grip], the current front of grips
            # paths_to_front: [paths_to_grip, ..., paths_to_grip]
            # paths_to_grip: [path_to_grip] or [path_to_grip_-1, path_to_grip_+1]
            # path_to_grip: ([*grip, module, ..., *grip], [-1 votes, +1 votes])
            def bfs_find_paths(front, paths_to_front):
                num_gvd = 0             # Number of grippers that have been visited
                while True:
                    # Update visiting status and path records based on current front
                    for i in range(len(front)):
                        grip = front[i]
                        if grip_needs_visiting[grip]:
                            grip_path[grip] = paths_to_front[i]
                            num_gvd = num_gvd + 1
                            if num_gvd >= num_gnv:
                                return
                        grip_visited[grip] = True
                    # Update front and paths to all grips in front
                    new_front = []
                    new_paths_to_front = []
                    grip_nf_idx = [-1] * (self.w + self.v)  # -1: Not in; >=0: The index
                    len_nf = 0
                    for i in range(len(front)):
                        grip = front[i]
                        for mdl_grip in self.get_grip_neighbors_w_mdl(grip):
                            if not grip_visited[mdl_grip[1]]:
                                if grip_nf_idx[mdl_grip[1]] < 0:
                                    grip_nf_idx.append(mdl_grip[1])
                                    new_front.append(mdl_grip[1])
                                    new_paths_to_front.append([])
                                    idx_nf = len_nf
                                    grip_nf_idx[mdl_grip[1]] = len_nf
                                    len_nf = len_nf + 1
                                else:
                                    idx_nf = grip_nf_idx[mdl_grip[1]]
                                for path_to_grip in paths_to_front[i]:
                                    path = path_to_grip[0]
                                    votes = [path_to_grip[1][0], path_to_grip[1][1]]
                                    len_path = len(path)
                                    if self.is_grip_w[path[len_path - 1]]:
                                        if len_path == 1:
                                            m1 = gf // 2
                                        else:
                                            m1= path[len_path - 2]
                                        grip = path[len_path - 1]
                                        m2 = mdl_grip[0]
                                        vote = self.get_polarity_vote(m1, grip, m2)
                                        if vote == -1:
                                            votes[0] = votes[0] + 1
                                        else:
                                            votes[1] = votes[1] + 1
                                    new_path = (path + mdl_grip, votes)
                                    if len(new_paths_to_front[idx_nf]) == 0:
                                        new_paths_to_front[idx_nf].append(new_path)
                                    elif len(new_paths_to_front[idx_nf]) == 1:
                                        new_paths_to_front[idx_nf].append(new_path)
                                        tendacy = new_paths_to_front[idx_nf][0][1][1] \
                                                - new_paths_to_front[idx_nf][0][1][0]
                                        new_tend = votes[1] - votes[0]
                                        if new_tend < tendacy:
                                            new_paths_to_front[idx_nf] = \
                                                new_paths_to_front[idx_nf][::-1]
                                    else:
                                        tend_0 = new_paths_to_front[idx_nf][0][1][0] \
                                                + new_paths_to_front[idx_nf][0][1][1]
                                        tend_1 = new_paths_to_front[idx_nf][1][1][0] \
                                                + new_paths_to_front[idx_nf][1][1][1]
                                        new_tend = votes[0] + votes[1]
                                        if new_tend < tend_0:
                                            new_paths_to_front[idx_nf][0] = new_path
                                        elif new_tend > tend_1:
                                            new_paths_to_front[idx_nf][1] = new_path
                    front = new_front
                    paths_to_front = new_paths_to_front
            if num_gnv > 0: bfs_find_paths([grip_start], [[([grip_start], [0, 0])]])
            for i in range(len(gps)):
                gps[i] = grip_path[grip_ends[i]]
                gt = gts[i]
                one_last_vote = False
                if self.module2gripper[gt % 2][gt // 2] < 0:
                    if self.is_grip_w[self.module2gripper[1 - gt % 2][gt // 2] // 3]:
                        one_last_vote = True
                for j in range(len(gps[i])):
                    gp = gps[i][j][0]
                    vnp = gps[i][j][1][0]   # Number of votes for negative polarity
                    vpp = gps[i][j][1][1]   # Number of votes for positive polarity
                    if one_last_vote:
                        if len(gp) <= 1:
                            m1 = gf // 2
                        else:
                            m1 = gp[-2]
                        grip = gp[-1]
                        m2 = gt // 2
                        vote = self.get_polarity_vote(m1, grip, m2)
                        if vote == -1:
                            vnp = vnp + 1
                        else:
                            vpp = vpp + 1
                    pv = (vnp, vpp)
                    if len(gps[i]) == 1:
                        actions.append((gf, gt, gp, -1, pv))
                        actions.append((gf, gt, gp, +1, pv))
                    else:
                        pp = 2 * j - 1
                        actions.append((gf, gt, gp, pp, pv))

        # DFS for finding directed bridges
        gripper_neighbors = [[]] * len(self.grippers)
        visited_time = [-1] * len(self.grippers)
        lowest_time = [-1] * len(self.grippers)
        for i in range(len(self.grippers)):
            gripper_neighbors[i] = self.get_gripper_neighbors(i)
        gripper_visited = [False] * len(self.grippers)
        gcr = [True] * len(self.grippers)               # Whether can grippers release
        def dfs_find_bridge(v, p, t, is_same_module):
            if not is_same_module:
                t = t + 1
            visited_time[v] = t
            lowest_time[v] = t
            gripper_visited[v] = True
            for to in gripper_neighbors[v]:
                if to == p:                             # If going back to parent
                    continue
                else:
                    if gripper_visited[to]:
                        lowest_time[v] = min(lowest_time[v], visited_time[to])
                    else:
                        b = self.gripper2module[v] // 2 == self.gripper2module[to] // 2
                        dfs_find_bridge(to, v, t, b)
                        lowest_time[v] = min(lowest_time[v], lowest_time[to])
                        if visited_time[v] < lowest_time[to]:
                            gcr[max(v, to)] = False     # This gripper can't release
        dfs_find_bridge(0, -1, 0, False)
        for grip in range(self.w + self.v):
            if self.is_grip_w[grip]:
                gripper = 3 * grip + 2
            else:
                gripper = 3 * grip + 1
            if gcr[gripper]:
                actions.append(self.gripper2module[gripper])
        return actions
    
    def execute_action(self, action):
        super().execute_action(action)
        if isinstance(action, tuple):
            self._execute_grasping(action[0], action[1], action[2], action[3], action[4])
        else:
            self._execute_releasing(action)

    def _execute_grasping(self, gf, gt, gp, pp, pv):
        pass

    def _execute_releasing(self, gb):
        pass
    
    def print_all_directions(self):
        print(f"Module Loops are: {self.module_loops}")
        print(f"Module Directions are: {self.module_ht_loops}")
        print(f"Grasp Loops are: {self.grasp_loops}")
        print(f"Grasp Directions are: {self.grasp_dir_loops}")

    def print_all_polarities(self):
        print(f"Module Polarities are: {self.module_polarities}")
        print(f"Loop Polarities are: {self.loop_polarities}")
        print(f"Grip Polarities are: {self.grip_polarities}")

    # Print Feasible Actions (Mainly for Testing and Debugging)
    def print_actions(self):
        print("Available Actions are: ")
        ht_str = ["Head", "Tail"]
        for action in self.actions:
            if isinstance(action, tuple):
                gf, gt, gp, pp, pv = action
                print(f"Module {gf // 2} {ht_str[gf % 2]} Grasps ", end = "")
                print(f"Module {gt // 2} {ht_str[gt % 2]} with: ")
                print(f"    path {gp}; polarity {pp} and votes {pv}")
            else:
                print(f"{ht_str[action % 2]} of Module {action // 2} Releases")

    @staticmethod
    def get_random_configuration(m, seed = None, w = None, v = None):
        params = EMRC.get_random_configuration_model(m ,seed, w, v)
        return EMRC(*params)