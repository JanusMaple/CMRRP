import numpy as np

from Module import Module
from GraspTheta import GraspTheta
from TMRC import TMRC
from GMRC import GMRC

class MRC:
    n_modules = 5

    def __init__(self, modules, gmrc):
        self.modules = modules
        for i in range(MRC.n_modules):
            self.modules[i].set_index(i)
            self.modules[i].rename("M" + str(i))
        
        self.gmrc = gmrc
    
    def __str__(self):
        output = ""
        for i in range(MRC.n_modules):
            description = self.modules[i].get_description()
            if not output == "" and not description == "":
                output = output + "\n"
            output = output + description
        return output
    
    def __repr__(self):
        return self.__str__()
    
    def get_actions(self):
        pass

    @staticmethod
    def Initiate(n_modules):
        MRC.n_modules = n_modules

    @staticmethod
    def get_simplest_configuration():
        modules = []
        for i in range(MRC.n_modules):
            modules.append(Module())
        
        for i in range(MRC.n_modules - 1):
            modules[i].catch(modules[i + 1], 0, 1)
        
        return MRC(modules)

    @staticmethod
    def get_random_configuration(seed = None, is_print = False, w = None, v = None):
        rng = np.random.default_rng(seed)
        # Get the total number of modules
        m = MRC.n_modules

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
        modules = []
        for i in range(m):
            modules.append(Module())
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

        # Random generate geometry from topology
        # NOTE: Only considering plane geometry for now
        gmrc = GMRC(w, v, n, m, grippers, gripper2module, module2gripper, rng)
        
        # Grasp process to form an MRC
        Module.start_hacking()
        idx_g = 0
        while idx_g < len(grippers):
            # If is the first gripper in grip or second gripper in w grip
            if (idx_g % 3 == 0) or ((idx_g % 3 == 1) and (grippers[idx_g + 1] > -2)):
                idx_m_catcher = gripper2module[idx_g] // 2
                ht_catcher = gripper2module[idx_g] % 2
                idx_m_catched = gripper2module[idx_g + 1] // 2
                ht_catched = gripper2module[idx_g + 1] % 2
                grsp_theta = GraspTheta(gmrc.grsp_angs[idx_g])
                modules[idx_m_catcher].catch(
                    modules[idx_m_catched], ht_catched, ht_catcher, grsp_theta)

                if idx_g % 3 == 0:
                    idx_g = idx_g + 1
                else:
                    idx_g = idx_g + 2
            elif (idx_g % 3 == 1) and (grippers[idx_g + 1] == -2):
                idx_g = idx_g + 2
            else:
                idx_g = idx_g + 1           # Should never happen though
        for idx_m in range(m):
            modules[idx_m].bend_to(gmrc.bend_angs[idx_m])
        Module.stop_hacking()

        if is_print:
            print("Module Robot Configuration Basic Data: ")
            print("w: ", end='', flush=True)
            print(w, end='; ', flush=True)
            print("v: ", end='', flush=True)
            print(v, end='; ', flush=True)
            print("n: ", end='', flush=True)
            print(n, end='; ', flush=True)
            print('Gripper state: ', end='', flush=True)
            print(grippers)
            print("-----------------------------------------------------------------")
            print('Module to grippers are: ', end='')
            print(gmrc.module2gripper)
            print("Grippers belong to module: ", end='')
            print([int(i // 2) for i in gmrc.gripper2module])
            print("Grippers' head-tail conditions are: ", end='')
            print([int(i % 2) for i in gmrc.gripper2module])
            print("-----------------------------------------------------------------")
            TMRC.show_topology(gmrc.G)
            print("Module Cycles are: ", end='')
            print(gmrc.mdl_cycles)
            print("Grip Cycles are: ", end='')
            print(gmrc.grip_cycles)
            print("Real Cycles are: ", end='')
            print(gmrc.real_cycles)
            print("-----------------------------------------------------------------")
            print("Module Loops are: ", end='')
            print(gmrc.module_loops)
            print("Module Directions are: ", end='')
            print(gmrc.module_ht_loops)
            print("Grasp Loops are: ", end='')
            print(gmrc.grasp_loops)
            print("Grasp Directions are: ", end='')
            print(gmrc.grasp_dir_loops)
            print("-----------------------------------------------------------------")
            print("Module Polarities are: ", end='')
            print(gmrc.module_polarities)
            print("Loop Polarities are: ", end='')
            print(gmrc.loop_polarities)
            print("Grip Polarities are: ", end='')
            print(gmrc.grip_polarities)
            gmrc.show_w_grip_modules()
            print("-----------------------------------------------------------------")
            print("x index to angles index mappings are ", end='')
            print(gmrc.xi2angi)
            print("Bending angle look-up indexes in x are ", end='')
            print(gmrc.ba_xi_loops)
            print("Grasping angle look-up indexes in x are ", end='')
            print(gmrc.ga_xi_loops)
            print("-----------------------------------------------------------------")
            print("Bending angles are ", end='')
            np.set_printoptions(precision=2, suppress=True, linewidth=1024)
            print(gmrc.bend_angs)
            print("Grasping angles are ", end='')
            print(gmrc.grsp_angs)
            print("-----------------------------------------------------------------")
            print("Available Actions are ", end='')
            print(gmrc.actions)
            gmrc.show_geometry()

        return MRC(modules, gmrc)
