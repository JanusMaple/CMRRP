from TMRC import TMRC

# Modular Robot Configuration with a Combinatorial Embedding
# 

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
    
    # Module_P: how a module is participating in a loop
    #   0: not in loop; 1: h-t in 360 loop or t-h in -360 loop; -1: otherwise
    # Loop_P: whether a loop is counterclockwise or clockwise by the list order
    #   0: to be decided; 1: 360 loop (counterclockwise); -1: -360 loop (clockwise)
    # Grip_P: the arrangement of the three grippers in a w-grip
    #   0: not in loop or is a v-grip
    #   1: gripper_1 -> gripper_2 -> gripper_3 in counterclockwise direction
    #   such that gamma_1 + gamma_2 + 180 ∈ [-115, 115] degrees
    #   -1: # 1: gripper_1 -> gripper_2 -> gripper_3 in clockwise direction
    #   such that gamma_1 + gamma_2 + 180 ∈ [245, 475] degrees
    #   Reason: gamma_1 + 180 + gamma_2 + 180 + gamma_3 + 180 = 360 (cc) or 720 (c)
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