from TMRC import TMRC

# Modular Robot Configuration with a Combinatorial Embedding
# The term "Polarity" is abused for representing:
#   1. The order of the edges in a cycle list
#   2. The rotation system of the grips
#   3. The orientation of modules in the cycles

class EMRC(TMRC):
    """
    The embedding of the configuration. 
    
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
        loop_polarities = [0] * self.c
        grip_polarities = [0] * (self.w + self.v)
        # Large cycles first
        for i in reversed(range(len(self.module_loops))):
            # Decide Loop Polarity
            module_loop = self.module_loops[i]
            for j in range(len(module_loop)):
                module = module_loop[j]
                # If the module already h-t in 360 or t-h in -360
                if module_polarities[module] == 1:
                    # Loop should be 360 for t-h or -360 for h-t
                    loop_polarities[i] = -self.module_ht_loops[i][j]
                    break
                elif module_polarities[module] == -1:
                    loop_polarities[i] = self.module_ht_loops[i][j]
                    break
            else:
                loop_polarities[i] = 1          # Default loop polarity is 360
            
            # Decide Module Polarity
            for j in range(len(module_loop)):   # Update module polarities
                module = module_loop[j]
                new_module_polarity = loop_polarities[i] * self.module_ht_loops[i][j]
                if module_polarities[module] == 0:
                    module_polarities[module] = new_module_polarity
                elif not module_polarities[module] == new_module_polarity:
                    module_polarities[module] = int(self.rng.choice([-1, 1], 1)[0])

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
    
    # Return: [(gf, gt, gp, sp), ..., (gf, gt, gp, sp), gb, ..., gb] 
    # gf: 2 * module + ht, the docking starts from this gripper
    # gt: 2 * module + ht, the docking goes to this gripper, which is be grasped by gf
    # gp: grip_path [grip, module, grip, ..., module, grip]
    #   The starting grip is the only neighbor of the leaf node gf
    #   The end grip is non-leaf node gt or the only neighbor of the leaf node gt
    # sp: suggested loop polarity (the probability for +1 polarity), sp ∈ (0, 1)
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
            grip_ends = [self.module2gripper[gt % 2][gt // 2] // 3 for gt in gts]
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
                    for i in range(len(front)):
                        grip = front[i]
                        for mdl_grip in self.get_grip_neighbors_w_mdl(grip):
                            if not grip_visited[mdl_grip[1]]:
                                new_front.append(mdl_grip[1])
                                new_path = paths_to_front[i] + mdl_grip
                                new_paths_to_front.append(new_path)
                    front = new_front
                    paths_to_front = new_paths_to_front
            if num_gnv > 0: bfs_find_paths([grip_start], [[grip_start]])
            for i in range(len(gps)):
                gps[i] = grip_path[grip_ends[i]]
                actions.append((gf, gts[i], gps[i]))

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
                actions.append(gripper)
        return actions