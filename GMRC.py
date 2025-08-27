import warnings
warnings.filterwarnings(
    "ignore",
    message="Values in x were outside bounds during a minimize step",
    category=RuntimeWarning
)
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from PIL import Image
import io
import copy
from scipy.optimize import minimize
class EarlyStop(Exception):
    def __init__(self, x, value):
        self.x = x
        self.value = value
from shapely import STRtree
from shapely.geometry import MultiLineString, LineString, box

from EMRC import EMRC

# Geometrical Modular Robot Configuration
# Convention: alpha: head starting angle; beta: bend angle; gamma: grasp angle
class GMRC(EMRC):
    mdl_ang_cap = np.pi         # 180 degree, must be in (0, 180)
    grsp_ang_cap = 2            # 115 degree, must be in (0, 180)

    radius_ratio = 1 / 10       # The body radius ratio for a module

    num_seg_lens = 5            # Number of line segments for a module
    _mdl_seg_lens = np.array([126.383, 58, 58, 58, 126.383], dtype=np.float64)
    mdl_seg_lens = _mdl_seg_lens / np.sum(_mdl_seg_lens)    # Five segments
    plg_axs_place =  np.array([0.77, 0.5, 0.5, 0.5, 0.23], dtype=np.float64)
    cls_exp_ratio = 0.01        # Collision exemption ratio

    drs_dis_thd = 0.01          # Dangerous Distance Threshold
    loop_ang_thd = np.pi / 6    # Loop angle optimization phase I threshold

    store_gif = False            # Whether to store a gif of action
    store_gif_filename = 'grasping_action.gif'

    # place (i, r): r position of segment i
    text_place = [(0, 0.3), (2, 0.5), (4, 0.7)]

    def __init__(self, w, v, n, m, grippers, gripper2module, module2gripper, rng,
                 loop_polarities = None, 
                 grip_polarities = None, 
                 bending_angles = None, 
                 grasping_angles = None): 
        super().__init__(w, v, n, m, grippers, gripper2module, module2gripper, rng, 
                         loop_polarities, grip_polarities)

        # NOTE: All ancillary properties are here to use for angle optimization
        # Number of different objects forming all loops
        self.number_module_in_loop = 0
        self.number_grasp_in_loop = 0   # Even the outgoing grasp attached to loop
        # Boundaries for x during optimization
        self.x_boundary = []    # A list of tuples of boundaries
        # From x index to bend_angs index or grsp_angs index
        self.xi2angi = []       # A list from x index to bend_angs or grsp_angs index
        # Relative properties for dealing with w-grip angle constraints
        self.xi_wf = []         # All the indexes for the first grasps of w-grip in x
        self.wgpa_obj = []      # w-grip angle objective (-180 for cc and 180 for c)
        # All these lists have elements of int arrays, serving as indexes for np.array
        self.ba_xi_loops = []   # A list of lists of indexes of x for bend_angs
        self.ga_xi_loops = []   # A list of lists of indexes of x for grsp_angs
        # All these lists have elements of np.array for accelerating optimizing
        self.bas_loops = []     # The bend angle sign for each bend angle in loop
        self.gas_loops = []     # The grasp angle sign for each grasp angle in loop

        # Initialize for gemetries and colliders
        # Format: {module_index: ((x, y), alpha, beta, ht), ...}
        self.module_geometries = dict.fromkeys(range(self.m))
        # Format: [(Bounding Box, MultiLineString), ...]
        self.module_colliders = [None] * self.m

        self.successfully_spawned = True

        if bending_angles is None or grasping_angles is None:
            is_planar, embedding = nx.check_planarity(self.G)   # Must be planar
            cannot_be_docked = False
            for try_collision_free in range(10):
                self.bend_angs, self.grsp_angs = self.get_random_angles_da()
                if cannot_be_docked:
                    print('\033[91mFailed for Docking Loops :(\033[0m')
                    self.successfully_spawned = False
                    break
                elif not is_planar:
                    print('\033[91mThe Graph is Not Planar :(\033[0m')
                    self.successfully_spawned = False
                    break
                if len(self.module_loops) > 0:              # If the graph is not acyclic
                    error = 1
                    for try_docking_loops in range(3):
                        x0 = self.initialize_x()            # Initialize all variables
                        x0 = self.optim_angles_la_x(x0)
                        x = self.optim_angles_ld_x(x0)
                        error = self.get_loop_dock_error_x(x)
                        if error > 1e-3:
                            self.bend_angs, self.grsp_angs = self.get_random_angles_da()
                        else:
                            break                           # Until global minimized
                    else:
                        cannot_be_docked = True
                    self.update_angs_from_x(x)
                self.update_all_module_geometry()           # Update all geometries
                self.update_all_module_collider()           # Update all colliders
                if not self.is_collision_detected():
                    break                                   # Until no collision
            else:                                           # Cannot break the loop
                print('\033[91mFailed for Finding Collision-Free Layout :(\033[0m')
                self.successfully_spawned = False
        else:
            self.bend_angs = bending_angles
            self.grsp_angs = grasping_angles
            self.update_all_module_geometry()               # Update all geometries
            self.update_all_module_collider()               # Update all colliders

    # Generate random angles under dock-angle constraint
    def get_random_angles_da(self):
        bend_angs = self.rng.uniform(-GMRC.mdl_ang_cap, GMRC.mdl_ang_cap, self.m)
        grsp_angs = np.zeros(len(self.grippers))
        for grip in range(self.w + self.v):
            if self.is_grip_w[grip]:
                p = self.grip_polarities[grip]  # Grip polarity decides grip type
                grsp_angs[3 * grip : 3 * (grip + 1)] = self.get_random_grip_angles(p)
            else:
                grsp_angs[3 * grip] = self.rng.uniform(
                    -GMRC.grsp_ang_cap, GMRC.grsp_ang_cap)
        return bend_angs, grsp_angs
    
    # Initialize x from angles (self.bend_angs and self.grsp_angs)
    def initialize_x(self):
        nmil = 0                                            # Number of Modules in Loop
        mi2xi = [-1] * self.m                               # Module idx to x idx
        self.ba_xi_loops = [np.zeros(len(module_loop), dtype=np.int64) 
                            for module_loop in self.module_loops]
        self.bas_loops = [np.array(module_ht_loop, dtype=np.float64) 
                          for module_ht_loop in self.module_ht_loops]
        for i in range(len(self.module_loops)):
            module_loop = self.module_loops[i]
            for j in range(len(module_loop)):
                module = module_loop[j]
                if mi2xi[module] < 0:
                    mi2xi[module] = nmil
                    self.ba_xi_loops[i][j] = nmil
                    nmil = nmil + 1
                else:
                    self.ba_xi_loops[i][j] = mi2xi[module]
        self.number_module_in_loop = nmil

        nwgil = 0
        nvgil = 0
        gi2xi = [-1] * (self.w + self.v)    # w-grip to the first grasp index out of 3
        self.ga_xi_loops = [np.zeros(len(grasp_loop), dtype=np.int64) 
                            for grasp_loop in self.grasp_loops]
        self.gas_loops = [np.array(grasp_dir_loop, dtype=np.float64) 
                          for grasp_dir_loop in self.grasp_dir_loops]
        for i in range(len(self.grasp_loops)):
            grasp_loop = self.grasp_loops[i]
            for j in range(len(grasp_loop)):
                grasp = grasp_loop[j]
                grip = grasp[0] // 3
                if gi2xi[grip] < 0:
                    gi2xi[grip] = nmil + 3 * nwgil + nvgil
                    if self.is_grip_w[grip]:
                        grasp_id = GMRC.grsp_identifier_2_id[
                            grasp[0] % 3 + grasp[1] % 3 - 1]
                        self.ga_xi_loops[i][j] = nmil + 3 * nwgil + nvgil + grasp_id
                        nwgil = nwgil + 1
                    else:
                        self.ga_xi_loops[i][j] = nmil + 3 * nwgil + nvgil
                        nvgil = nvgil + 1
                else:
                    if self.is_grip_w[grip]:
                        grasp_id = GMRC.grsp_identifier_2_id[
                            grasp[0] % 3 + grasp[1] % 3 - 1]
                        self.ga_xi_loops[i][j] = gi2xi[grip] + grasp_id
                    else:
                        self.ga_xi_loops[i][j] = gi2xi[grip]
        self.number_grasp_in_loop = 3 * nwgil + nvgil

        self.x_boundary = [(-GMRC.mdl_ang_cap, GMRC.mdl_ang_cap)] \
            * self.number_module_in_loop
        self.x_boundary.extend([(-GMRC.grsp_ang_cap, GMRC.grsp_ang_cap)] \
                               * self.number_grasp_in_loop)

        x0 = np.zeros(nmil + 3 * nwgil + nvgil, dtype=np.float64)
        self.xi2angi = np.zeros(nmil + 3 * nwgil + nvgil, dtype=np.int64)
        for i in range(len(mi2xi)):
            if mi2xi[i] >= 0:
                x0[mi2xi[i]] = self.bend_angs[i]
                self.xi2angi[mi2xi[i]] = i
        self.xi_wf = []
        self.wgpa_obj = []
        for i in range(len(gi2xi)):             # i is the grip
            if gi2xi[i] >= 0:
                if self.is_grip_w[i]:
                    self.xi_wf.append(gi2xi[i])
                    # sum + 3 * 180 = 360 or 720 => sum = -180 or 180
                    self.wgpa_obj.append(-self.grip_polarities[i] * np.pi)
                    for j in range(3):
                        x0[gi2xi[i] + j] = self.grsp_angs[3 * i + j]
                        self.xi2angi[gi2xi[i] + j] = 3 * i + j
                else:
                    x0[gi2xi[i]] = self.grsp_angs[3 * i]
                    self.xi2angi[gi2xi[i]] = 3 * i
        return x0
    
    # Optimize angles in x to minimize loop-angle and w_grip_angle error
    def optim_angles_la_x(self, x0):
        if len(self.xi_wf) > 0:
            constraints = [
                {
                    'type': 'eq', 
                    'fun': self.get_w_grip_angle_error_x
                }
            ]
        else:
            constraints = []
        try:
            result = minimize(
                self.get_loop_angle_error_x,
                x0,
                args=(True),
                method='SLSQP',
                constraints=constraints,
                bounds=self.x_boundary)
            return result.x
        except EarlyStop as e:
            result = {
                'x': e.x,
                'fun': e.value,
                'message': 'Optmization Stopped Early',
                'success': True,
                'status': 0
            }
            return result['x']
    
    # Optimize angles in x to minimize loop-dock error
    def optim_angles_ld_x(self, x0):
        if len(self.xi_wf) > 0:
            constraints = [
                {
                    'type': 'eq', 
                    'fun': self.get_w_grip_angle_error_x
                }, 
                {
                    'type': 'eq', 
                    'fun': self.get_loop_angle_error_x,
                    'args': (False, )
                }
            ]
        else:
            constraints = [
                {
                    'type': 'eq', 
                    'fun': self.get_loop_angle_error_x,
                    'args': (False, )
                }
            ]
        try:
            result = minimize(
                self.get_loop_dock_error_x,
                x0,
                args=(True, ),
                method='SLSQP',
                constraints=constraints,
                bounds=self.x_boundary)
            return result.x
        except EarlyStop as e:
            result = {
                'x': e.x,
                'fun': e.value,
                'message': 'Optmization Stopped Early',
                'success': True,
                'status': 0
            }
            return result['x']
    
    # Get w-grip angle error (from 360 or 720 depending on the direction)
    def get_w_grip_angle_error_x(self, x):
        errors = np.zeros(len(self.xi_wf))
        for i in range(len(self.xi_wf)):
            j = self.xi_wf[i]
            errors[i] = np.abs(x[j] + x[j + 1] + x[j + 2] - self.wgpa_obj[i])
        return errors

    # Get loop-angle error for x
    def get_loop_angle_error_x(self, x, is_obj = False):
        errors = np.zeros(len(self.ba_xi_loops))
        total_error = 0
        for i in range(len(self.ba_xi_loops)):
            errors[i] = (np.sum(x[self.ba_xi_loops[i]] * self.bas_loops[i])
                + np.sum(x[self.ga_xi_loops[i]] * self.gas_loops[i])
                - self.loop_polarities[i] * 2 * np.pi)
            total_error = total_error + errors[i]
        if is_obj and total_error <= 1e-5:
            raise EarlyStop(x, total_error)
        if is_obj:
            return total_error
        else:
            return errors

    # Get loop-dock error for x
    def get_loop_dock_error_x(self, x, is_obj = False):
        error = 0
        for i in range(len(self.ba_xi_loops)):
            betas = x[self.ba_xi_loops[i]] * self.bas_loops[i]
            gammas = x[self.ga_xi_loops[i]] * self.gas_loops[i]
            l = len(self.ba_xi_loops[i])
            error = error + GMRC.get_single_loop_dock_error(betas, gammas, l)
        if is_obj and error <= 1e-5:
            raise EarlyStop(x, error)
        return error
    
    # Update angles (self.bend_angs and self.grsp_angs) from x
    def update_angs_from_x(self, x):
        for i in range(self.number_module_in_loop):
            self.bend_angs[self.xi2angi[i]] = x[i]
        for i in range(self.number_module_in_loop, len(x)):
            self.grsp_angs[self.xi2angi[i]] = x[i]

    # Generate random angles for a w-grip with polarity
    def get_random_grip_angles(self, polarity):
        # Range of the sum of gamma1 and gamma2
        gamma_sum_min = -polarity * np.pi - GMRC.grsp_ang_cap
        gamma_sum_max = -polarity * np.pi + GMRC.grsp_ang_cap

        gamma1_min = max(gamma_sum_min - GMRC.grsp_ang_cap, -GMRC.grsp_ang_cap)
        gamma1_max = min(gamma_sum_max + GMRC.grsp_ang_cap, GMRC.grsp_ang_cap)
        gamma1 = self.rng.uniform(gamma1_min, gamma1_max)

        gamma2_min = max(gamma_sum_min - gamma1, -GMRC.grsp_ang_cap)
        gamma2_max = min(gamma_sum_max - gamma1, GMRC.grsp_ang_cap)
        gamma2 = self.rng.uniform(gamma2_min, gamma2_max)
        
        gamma3 = -polarity * np.pi - gamma1 - gamma2

        return [gamma1, gamma2, gamma3]
    
    # Generate random angle for the new grasping angles of a w-grip from v-grip
    def get_gamma2_range(self, gamma1, polarity):
        gamma_sum_min = -polarity * np.pi - GMRC.grsp_ang_cap
        gamma_sum_max = -polarity * np.pi + GMRC.grsp_ang_cap
        
        gamma2_min = max(gamma_sum_min - gamma1, -GMRC.grsp_ang_cap)
        gamma2_max = min(gamma_sum_max - gamma1, GMRC.grsp_ang_cap)

        return (gamma2_min, gamma2_max)

    # Get grasp angle from gripper_1 to gripper_2
    def get_grasp_angle(self, g1, g2):
        assert ((g1 // 3 == g2 // 3) and not (g1 == g2))
        grip = g1 // 3
        if self.is_grip_w[grip]:
            # 0 -> 1, 1 -> 2, 2 -> 0
            if (g1 + 1) % 3 == g2 % 3:
                return self.grsp_angs[g1]
            # 1 -> 0, 2 -> 1, 0 -> 2
            else:
                return -self.grsp_angs[g2]
        else:
            # 0 -> 1
            if g1 < g2:
                return self.grsp_angs[g1]
            else:
                return -self.grsp_angs[g2]
            
    # Update geometries of all modules
    def update_all_module_geometry(self, ng = None):
        self.mdl_geo_updated = [False] * self.m
        self._update_module_geometry(0, (0, 0), 0, 0, ng)

    # Update all module colliders
    def update_all_module_collider(self):
        for i in range(self.m):
            self.module_colliders[i] = GMRC.get_module_collider(
                self.module_geometries[i])

    # Update geometry recursively
    # mi: module index; sp: starting point; sa: starting angle
    # ht: head or tail that starts; ng: neglected gripper
    def _update_module_geometry(self, mi, sp, sa, ht, ng = None):
        if ht == 0:
            ar = self.bend_angs[mi]
        else:
            ar = -self.bend_angs[mi]

        angs, xy = GMRC.get_mdl_seg_geo(sp, sa, ar)
        ep = (xy[GMRC.num_seg_lens, 0], xy[GMRC.num_seg_lens, 1])
        ea = angs[len(angs) - 1]

        if ht == 0:
            ht_position = (sp, ep)                  # Head and tail positions
            ht_angle = (sa + np.pi, ea)             # Head and tail angles
        else:
            ht_position = (ep, sp)
            ht_angle = (ea, sa + np.pi)

        self.module_geometries[int(mi)] = (
            ht_position[0], 
            ht_angle[0] - np.pi,                    # Growing angle is opposite to it 
            self.bend_angs[mi],
            )
        self.mdl_geo_updated[mi] = True

        for i in range(2):
            if self.module2gripper[i][mi] >= 0:     # If it participates in a grip
                gripper = self.module2gripper[i][mi]
                gns = self.get_gripper_grip_neighbors(gripper)  # Gripper Grip Neighbors
                for gn in gns:
                    if ng is not None:                          # If neglecting grasp
                        if gripper == ng or gn == ng:
                            continue
                    mn = self.gripper2module[gn] // 2           # Module Neighbor
                    mn_ht = self.gripper2module[gn] % 2
                    if not self.mdl_geo_updated[mn]:
                        mn_sp = ht_position[i]
                        mn_sa = ht_angle[i] + self.get_grasp_angle(gripper, gn)
                        self._update_module_geometry(mn, mn_sp, mn_sa, mn_ht, ng)

    # Detect collision between all modules
    def is_collision_detected(self):
        bbxs = []
        for i in range(self.m):
            bbxs.append(self.module_colliders[i][0])
        tree = STRtree(np.array(bbxs))
        for i in range(self.m - 1):
            neighbors = tree.query_nearest(self.module_colliders[i][0], exclusive=True)
            for n in neighbors:
                if self.module_colliders[i][1].intersects(self.module_colliders[n][1]):
                    return True
        return False
    
    # Get dangerous distance of the current configuration
    # Return: float ∈ [-GMRC.drs_dis_thd, +∞); Better to be larger than zero
    def get_dangerous_distance(self):
        # dd_colliders[i]: (mls, hls, tls)
        dd_colliders = self._get_dd_colliders()
        min_dis = 1e6
        for i in range(self.m):
            for j in range(i + 1, self.m):
                gih = self.module2gripper[0][i]             # Gripper of module i head
                git = self.module2gripper[1][i]             # Gripper of module i tail
                gjh = self.module2gripper[0][j]             # Gripper of module j head
                gjt = self.module2gripper[1][j]             # Gripper of module j tail
                if gih >= 0 and gjh >= 0 and gih // 3 == gjh // 3:
                    if git >= 0 and gjt >= 0 and git // 3 == gjt // 3:
                        continue
                    distance = dd_colliders[i][2].distance(dd_colliders[j][2])
                elif git >= 0 and gjh >= 0 and git // 3 == gjh // 3:
                    if gih >= 0 and gjt >= 0 and gih // 3 == gjt // 3:
                        continue
                    distance = dd_colliders[i][1].distance(dd_colliders[j][2])
                elif gih >= 0 and gjt >= 0 and gih // 3 == gjt // 3:
                    if git >= 0 and gjh >= 0 and git // 3 == gjh // 3:
                        continue
                    distance = dd_colliders[i][2].distance(dd_colliders[j][1])
                elif git >= 0 and gjt >= 0 and git // 3 == gjt // 3:
                    if gih >= 0 and gjh >= 0 and gih // 3 == gjh // 3:
                        continue
                    distance = dd_colliders[i][1].distance(dd_colliders[j][1])
                else:
                    distance = dd_colliders[i][0].distance(dd_colliders[j][0])
                if distance < min_dis:
                    min_dis = distance
                    if min_dis <= 0.0:
                        break
        return min_dis

    # get necessary colliders for calculating dangerous distance
    def _get_dd_colliders(self):
        dd_colliders = [None] * self.m
        for i in range(self.m):
            geometry = self.module_geometries[i]
            dd_colliders[i] = GMRC._get_dd_collider(geometry)
        return dd_colliders

    # Get all actions using emrc.get_all_actions()
    # Delete all actions that conflict grasping angle constraints
    def get_all_actions(self):
        actions = super().get_all_actions()
        act_idx_to_del = []
        for i in range(len(actions)):
            action = actions[i]
            if isinstance(action, tuple):                       # If grasp
                gt = action[1]
                if self.module2gripper[gt % 2][gt // 2] >= 0:   # If grasp a v-grip
                    path_polarity = action[3]
                    grsped_gripper = self.get_gripper(action[2][-2], action[2][-1])
                    if grsped_gripper % 3 == 0:
                        grip_polarity = path_polarity
                    else:
                        grip_polarity = -path_polarity
                    gamma_v = self.grsp_angs[3 * action[2][-1]]
                    gamma_range = self.get_gamma2_range(gamma_v, grip_polarity)
                    if gamma_range[0] > gamma_range[1]:
                        act_idx_to_del = [i] + act_idx_to_del   # Inverse order
        for i in act_idx_to_del:
            actions[i : i + 1] = []
        return actions

    # angle: A specified grasping angle for the action
    # To update: bend_angs, grsp_angs, module_geometries, module_colliders
    def execute_action(self, action, angle = None):
        grip_status = super().execute_action(action)
        if isinstance(action, tuple):
            result = GMRC._execute_grasping(self, grip_status, angle)
        else:
            GMRC._execute_releasing(self, grip_status)
            result = True
        return result

    # If gamma is None or out of boundary, then feel free to optimize gamma
    def _execute_grasping(self, grip_status, gamma = None):
        y0, is_optim_gamma = self.initialize_y(grip_status, gamma)

        if GMRC.store_gif:
            fig, self.gif_ax = plt.subplots()
            self.gif_frames = []

        y = self.optim_angles_y_ang(y0, is_optim_gamma, grip_status)
        y, error = self.optim_angles_y_all(y, is_optim_gamma, grip_status)

        if GMRC.store_gif:
            self.gif_frames[0].save(
                self.store_gif_filename,
                format='GIF',
                save_all=True,
                append_images=self.gif_frames[1:],
                duration=200,               # Duration between frames in milliseconds
                loop=0                      # Loop indefinitely
            )

        self.bend_angs = y[0 : self.m]                                  # bend_angs
        if is_optim_gamma:
            self._update_grsp_angs_from_gamma(y[-1], grip_status)       # grsp_angs
        if grip_status[1] == 0:                 # v-grip to w-grip
            ng = 3 * grip_status[0] + 2
        else:                                   # Created v-grip
            ng = 3 * grip_status[0] + 1
        self.update_all_module_geometry(ng)                             # Geometries
        self.update_all_module_collider()                               # Colliders
        
        is_success = True
        if error > 1e-3 or self.is_collision_detected():
            print("\033[91mAction Failed\033[0m: Failed to dock the new loop!")
            is_success = False
        return is_success
    
    # Try to modify the grasping angle of the outer gripper of a grip
    # grip: the grip to be modified; ang: the angle target of the grapsing angle
    # any_ang: whether is modifying any angle or new formed angle from grasping action
    def modify_grsp_ang(self, grip, ang, any_ang = False):
        if self.is_grip_w[grip]:
            grip_status = (grip, 0)         # Pretend to grasp v-grip and form w-grip
            gamma_0 = self.grsp_angs[3 * grip + 1]
            ng = 3 * grip + 2
        else:
            grip_status = (grip, 1)         # Pretend to grasp leaf and form v-grip
            gamma_0 = self.grsp_angs[3 * grip]
            ng = 3 * grip_status[0] + 1
        y0, _ = self.initialize_y(grip_status, gamma_0, mdf_mode=True)
        y, error = self.optim_angles_y_modify(y0, grip_status, ang)
        self.bend_angs = y[0 : self.m]
        self._update_grsp_angs_from_gamma(y[-1], grip_status)
        self.update_all_module_geometry(ng)
        self.update_all_module_collider()
        if error > 1e-3 or self.is_collision_detected():
            y, error = self.optim_angles_y_modify(y0, grip_status, ang, True)
            self.bend_angs = y[0 : self.m]
            self._update_grsp_angs_from_gamma(y[-1], grip_status)
            self.update_all_module_geometry(ng)
            self.update_all_module_collider()
            if error > 1e-3 or self.is_collision_detected():
                print("\033[91mAttempt Failed\033[0m: Failed to modify grasp angle!")
                return False
            else:
                if any_ang:
                    self.update_all_module_geometry()
                    self.update_all_module_collider()
                return True
        else:
            if any_ang:
                self.update_all_module_geometry()
                self.update_all_module_collider()
            return True

    # Initialize y for optimize the docking of a grasping action
    # mdf_mode: Whether in modifying graping angle mode or in docking mode
    def initialize_y(self, grip_status, gamma, mdf_mode = False):
        if grip_status[1] == 1:     # Created grip_status[0]
            gamma_range = (-GMRC.grsp_ang_cap, GMRC.grsp_ang_cap)
            if not mdf_mode:
                self.grsp_angs = np.append(self.grsp_angs, [0.0, 0.0, 0.0])
        else:                       # v -> w for grip_status[0]
            gamma1 = self.grsp_angs[3 * grip_status[0]]
            polarity = self.grip_polarities[grip_status[0]]
            gamma_range = self.get_gamma2_range(gamma1, polarity)
        
        if mdf_mode:                # No need to change gamma
            is_optim_gamma = True
        elif gamma is not None and gamma > gamma_range[0] and gamma < gamma_range[1]:
            self._update_grsp_angs_from_gamma(gamma, grip_status)
            is_optim_gamma = False
        else:
            gamma = self.rng.uniform(gamma_range[0], gamma_range[1])
            is_optim_gamma = True

        self.ba_yi_loops = [np.array(module_loop, dtype=np.int64) 
                            for module_loop in self.module_loops]
        self.bas_loops = [np.array(module_ht_loop, dtype=np.float64) 
                          for module_ht_loop in self.module_ht_loops]

        self.ga_gi_loops = [np.zeros(len(grasp_loop), dtype=np.int64) 
                            for grasp_loop in self.grasp_loops]
        self.gas_loops = [np.array(grasp_dir_loop, dtype=np.float64) 
                          for grasp_dir_loop in self.grasp_dir_loops]
        for i in range(len(self.grasp_loops)):
            grasp_loop = self.grasp_loops[i]
            for j in range(len(grasp_loop)):
                grasp = grasp_loop[j]
                grip = grasp[0] // 3
                grasp_id = GMRC.grsp_identifier_2_id[grasp[0] % 3 + grasp[1] % 3 - 1]
                self.ga_gi_loops[i][j] = 3 * grip + grasp_id

        self.y_boundary = [(-GMRC.mdl_ang_cap, GMRC.mdl_ang_cap)] * self.m

        if is_optim_gamma:
            y0 = np.zeros(self.m + 1)
            y0[0 : self.m] = self.bend_angs
            y0[-1] = gamma
            self.y_boundary.append(gamma_range)
        else:
            y0 = np.zeros(self.m)
            y0[:] = self.bend_angs
        return (y0, is_optim_gamma)
    
    # Get the out gripper grasping angle of a certain grip
    def get_grip_gamma(self, grip):
        if self.is_grip_w:
            return self.grsp_angs[3 * grip + 1]
        else:
            return self.grsp_angs[3 * grip]
        
    # Optimize y for minimizing loop angle errors
    def optim_angles_y_ang(self, y0, is_optim_gamma = False, grip_status = None):
        if self.c <= 1:
            constraints = [
                {
                    'type': 'ineq',
                    'fun': self.get_module_collision_error_y,
                    'args': (is_optim_gamma, grip_status)
                }
            ]
        else:
            constraints = [{
                    'type': 'eq',
                    'fun': self.get_loop_error_con_all_y,
            },
                {
                    'type': 'ineq',
                    'fun': self.get_module_collision_error_y,
                    'args': (is_optim_gamma, grip_status)
                }
            ]
        try:
            result = minimize(
                self.get_loop_error_obj_ang_y,
                y0,
                args=(is_optim_gamma, grip_status),
                method = 'SLSQP',
                constraints=constraints,
                bounds=self.y_boundary,
                callback=self.optim_y_callback,
                options={'eps': 1e-6, 'disp': False}
            )
            return result.x
        except EarlyStop as e:
            result = {
                'x': e.x,
                'fun': e.value,
                'message': 'Optmization Stopped Early',
                'success': True,
                'status': 0
            }
            return result['x']

    # Optimize y for both angle aligning and docking when 
    def optim_angles_y_all(self, y0, is_optim_gamma = False, grip_status = None):
        if self.c <= 1:
            constraints = [
                {
                    'type': 'ineq',
                    'fun': self.get_module_collision_error_y,
                    'args': (is_optim_gamma, grip_status)
                }
            ]
        else:
            constraints = [{
                    'type': 'eq',
                    'fun': self.get_loop_error_con_all_y,
            },
                {
                    'type': 'ineq',
                    'fun': self.get_module_collision_error_y,
                    'args': (is_optim_gamma, grip_status)
                }
            ]
        try:
            result = minimize(
                self.get_loop_error_obj_all_y,
                y0,
                args=(is_optim_gamma, grip_status),
                method = 'SLSQP',
                constraints=constraints,
                bounds=self.y_boundary,
                callback=self.optim_y_callback,
                options={'eps': 1e-6, 'disp': False}
            )
            return (result.x, result.fun)
        except EarlyStop as e:
            result = {
                'x': e.x,
                'fun': e.value,
                'message': 'Optmization Stopped Early',
                'success': True,
                'status': 0
            }
            return (result['x'], result['fun'])
        
    # Optimize y for modifying the grasping angle
    def optim_angles_y_modify(self, y0, grip_status, gamma_target, accurate = False):
        constraints = [{
            'type': 'eq',
            'fun': self.get_loop_error_con_mdf_y,
            'args': (grip_status, )
        },
        {
            'type': 'ineq',
            'fun': self.get_module_collision_error_y,
            'args': (True, grip_status)
        }]
        if accurate:
            method = 'trust-constr'
        else:
            method = 'SLSQP'
        result = minimize(
            self.get_gamma_modifying_error_y,
            y0,
            args=(gamma_target),
            method = method,
            constraints=constraints,
            bounds=self.y_boundary,
            options={'disp': False}
        )
        errors = self.get_loop_error_con_mdf_y(result.x, grip_status)
        error = np.linalg.norm(errors)
        return (result.x, error)

    # Optimization callback function for storing gif of the action
    def optim_y_callback(self, y_k):
         if GMRC.store_gif:
            # NOTE: # Geomery has already been updated by get_module_collision_error_y
            self.update_all_module_collider()
            self.gif_ax.clear()
            self.gif_ax.set_aspect('equal')
            self.gif_ax.axis('off')
            # self.gif_ax.set_xlim(-self.m * 1.0, self.m * 1.0)
            # self.gif_ax.set_ylim(-self.m * 1.0, self.m * 1.0)
            for i in range(self.m):
                g1n = f"H{self.module2gripper[0][i]}"
                mn = f"{i}"
                g2n = f"T{self.module2gripper[1][i]}"
                GMRC.draw_module(
                    self.gif_ax, 
                    self.module_geometries[i][0], 
                    self.module_geometries[i][1], 
                    self.module_geometries[i][2], 
                    g1n, 
                    mn, 
                    g2n
                )
                for line in self.module_colliders[i][1].geoms:
                    x, y = line.xy
                    self.gif_ax.plot(x, y, color = 'b')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            self.gif_frames.append(Image.open(buf))

    # Get module collision error objective for given y
    def get_module_collision_error_y(self, y, is_optim_gamma, grip_status):
        if is_optim_gamma:
            self._update_grsp_angs_from_gamma(y[-1], grip_status)
        self.bend_angs = y[0 : self.m]
        if grip_status[1] == 0:                 # v-grip to w-grip
            ng = 3 * grip_status[0] + 2
        else:                                   # Created v-grip
            ng = 3 * grip_status[0] + 1
        self.update_all_module_geometry(ng)
        dd =  self.get_dangerous_distance()
        dd_error = min(dd - GMRC.drs_dis_thd, GMRC.drs_dis_thd) / GMRC.drs_dis_thd
        return dd_error

    # Get a formed loop error condiering both loop angle and loop dock for y
    def get_loop_error_con_all_y(self, y):                                  # Constraint
        loop_angle_error = 0
        loop_dock_error = 0
        for i in range(self.c - 1):
            betas = y[self.ba_yi_loops[i]] * self.bas_loops[i]
            gammas = self.grsp_angs[self.ga_gi_loops[i]] * self.gas_loops[i]
            ang_sum_tar = self.loop_polarities[i] * 2 * np.pi
            loop_length = len(self.ba_yi_loops[i])
            loop_angle_error = loop_angle_error + \
                np.abs(np.sum(betas) + np.sum(gammas) - ang_sum_tar)
            loop_dock_error = loop_dock_error + \
                GMRC.get_single_loop_dock_error(betas, gammas, loop_length)
        error = loop_angle_error + loop_dock_error
        return error * 100.0

    # Get the last loop error condiering both loop angle and loop docking for y
    def get_loop_error_obj_all_y(self, y, is_optim_gamma, grip_status):     # Objective
        if is_optim_gamma:
            self._update_grsp_angs_from_gamma(y[-1], grip_status)
        betas = y[self.ba_yi_loops[-1]] * self.bas_loops[-1]
        gammas = self.grsp_angs[self.ga_gi_loops[-1]] * self.gas_loops[-1]
        ang_sum_tar = self.loop_polarities[-1] * 2 * np.pi
        loop_length = len(self.ba_yi_loops[-1])
        loop_angle_error = np.abs(np.sum(betas) + np.sum(gammas) - ang_sum_tar)
        loop_dock_error = GMRC.get_single_loop_dock_error(betas, gammas, loop_length)
        error = loop_angle_error + loop_dock_error
        if loop_angle_error <= 1e-5 and loop_dock_error <= 1e-5:
            raise EarlyStop(y, error)
        return error
    
    # Get the last loop error condiering only loop angle
    def get_loop_error_obj_ang_y(self, y, is_optim_gamma, grip_status):     # Objective
        if is_optim_gamma:
            self._update_grsp_angs_from_gamma(y[-1], grip_status)
        betas = y[self.ba_yi_loops[-1]] * self.bas_loops[-1]
        gammas = self.grsp_angs[self.ga_gi_loops[-1]] * self.gas_loops[-1]
        ang_sum_tar = self.loop_polarities[-1] * 2 * np.pi
        error = np.abs(np.sum(betas) + np.sum(gammas) - ang_sum_tar)
        if error <= GMRC.loop_ang_thd:                                      # Very Loose
            raise EarlyStop(y, error)
        return error
    
    # Get all loops error after modifying the grasping angle and bending angles
    def get_loop_error_con_mdf_y(self, y, grip_status):                     # Constraint
        errors = np.zeros(self.c * 3)
        self._update_grsp_angs_from_gamma(y[-1], grip_status)
        for i in range(self.c):
            betas = y[self.ba_yi_loops[i]] * self.bas_loops[i]
            gammas = self.grsp_angs[self.ga_gi_loops[i]] * self.gas_loops[i]
            ang_sum_tar = self.loop_polarities[i] * 2 * np.pi
            loop_length = len(self.ba_yi_loops[i])
            dtheta = np.sum(betas) + np.sum(gammas) - ang_sum_tar
            dx, dy = GMRC.get_single_loop_dock_error(betas, gammas, loop_length, True)
            errors[3 * i] = dtheta
            errors[3 * i + 1] = dx
            errors[3 * i + 2] = dy
        return np.array(errors)

    # Get the gamma error when modifying the grasping angle to approach a certain target
    def get_gamma_modifying_error_y(self, y, gamma_target):
        return np.abs(y[-1] - gamma_target)

    # Update self.grsp_angles from an angle gamma and grip_status
    def _update_grsp_angs_from_gamma(self, gamma, grip_status):
        if grip_status[1] == 1:
            self.grsp_angs[3 * grip_status[0]] = gamma
        else:
            self.grsp_angs[3 * grip_status[0] + 1] = gamma
            self.grsp_angs[3 * grip_status[0] + 2] = \
                -self.grip_polarities[grip_status[0]] * np.pi \
                -self.grsp_angs[3 * grip_status[0]] - gamma

    def _execute_releasing(self, grip_status):
        if grip_status[1] == -1:    # Deleted grip_status[0]
            self.grsp_angs = np.delete(self.grsp_angs, 
                      list(range(3 * grip_status[0], 3 * grip_status[0] + 3)))
        else:                       # w -> v for grip_status[0]
            self.grsp_angs[3 * grip_status[0] + 1] = 0
            self.grsp_angs[3 * grip_status[0] + 2] = 0

    # Get set \Gamma_{final} for discretizing grasping angle choices
    def get_Gamma_final(self):
        Gamma_final = []        # List of tuples: (ang, g1, g2, index, is_w_grip)

        for grip in range(self.w + self.v):
            if self.is_grip_w[grip]:
                Gamma_final.append((self.grsp_angs[3 * grip],
                                   self.gripper2module[3 * grip + 1],
                                   self.gripper2module[3 * grip],
                                   0, True))
                Gamma_final.append((-self.grsp_angs[3 * grip],
                                   self.gripper2module[3 * grip],
                                   self.gripper2module[3 * grip + 1],
                                   1, True))
                Gamma_final.append((self.grsp_angs[3 * grip + 1],
                                   self.gripper2module[3 * grip + 2],
                                   self.gripper2module[3 * grip + 1],
                                   2, True))
                Gamma_final.append((-self.grsp_angs[3 * grip + 1],
                                   self.gripper2module[3 * grip + 1],
                                   self.gripper2module[3 * grip + 2],
                                   3, True))
                Gamma_final.append((self.grsp_angs[3 * grip + 2],
                                   self.gripper2module[3 * grip],
                                   self.gripper2module[3 * grip + 2],
                                   4, True))
                Gamma_final.append((-self.grsp_angs[3 * grip + 2],
                                   self.gripper2module[3 * grip + 2],
                                   self.gripper2module[3 * grip],
                                   5, True))
            else:
                Gamma_final.append((self.grsp_angs[3 * grip],
                                   self.gripper2module[3 * grip + 1],
                                   self.gripper2module[3 * grip],
                                   0, False))
                Gamma_final.append((-self.grsp_angs[3 * grip],
                                   self.gripper2module[3 * grip],
                                   self.gripper2module[3 * grip + 1],
                                   1, False))

        return Gamma_final

    def print_all_angs(self):
        bend_ang_str = "Bend Angles: ["
        for ang in self.bend_angs:
            bend_ang_str = bend_ang_str + f"{round(float(ang * 180 / np.pi))}° "
        bend_ang_str = bend_ang_str + "]"
        print(bend_ang_str)
        grsp_ang_str = "Grasp Angles: ["
        for ang in self.grsp_angs:
            grsp_ang_str = grsp_ang_str + f"{round(float(ang * 180 / np.pi))}° "
        grsp_ang_str = grsp_ang_str + "]"
        print(grsp_ang_str)

    def show_all(self):
        self.print_all()
        self.show_topology()
        self.show_geometry()

    def show_geometry(self, simple = False):
        
        if self.module_geometries == {}:
            return

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.axis('off')

        leaf_count = self.w + self.v

        for i in range(self.m):
            g1n = f"H{self.module2gripper[0][i]}"
            mn = f"{i}"
            g2n = f"T{self.module2gripper[1][i]}"
            h_node = None
            t_node = None
            if self.module2gripper[0][i] % 3 == 0:
                h_node = self.module2gripper[0][i] // 3
            elif self.module2gripper[0][i] < 0:
                h_node = leaf_count
                leaf_count = leaf_count + 1
            if self.module2gripper[1][i] % 3 == 0:
                t_node = self.module2gripper[1][i] // 3
            elif self.module2gripper[1][i] < 0:
                t_node = leaf_count
                leaf_count = leaf_count + 1
            GMRC.draw_module(
                ax, 
                self.module_geometries[i][0], 
                self.module_geometries[i][1], 
                self.module_geometries[i][2], 
                g1n, 
                mn, 
                g2n,
                simple,
                h_node,
                t_node
            )
            for line in self.module_colliders[i][1].geoms:
                x, y = line.xy
                ax.plot(x, y, color = 'dimgray')

    def print_all(self):
        print("-----------------------------------------------------------------")
        self.print_configuration_data()
        print("-----------------------------------------------------------------")
        self.print_mdl_gpr_mapping()
        print("-----------------------------------------------------------------")
        self.print_all_cycles()
        print("-----------------------------------------------------------------")
        self.print_all_directions()
        print("-----------------------------------------------------------------")
        self.print_all_polarities()
        print("-----------------------------------------------------------------")
        self.print_all_angs()

    def copy(self):
        return copy.deepcopy(self)

    # Get loop-dock error for a single loop
    @staticmethod
    def get_single_loop_dock_error(betas, gammas, l, as_tuple = False):
        a = np.zeros((l, 1))
        b = np.zeros((l, 1))
        for j in range(GMRC.num_seg_lens):
            cur_betas = betas / (GMRC.num_seg_lens - 1) * j
            a = a + np.cos(cur_betas) * GMRC.mdl_seg_lens[j]
            b = b + np.sin(cur_betas) * GMRC.mdl_seg_lens[j]
        
        beta_cml = np.cumsum(betas)
        beta_cml = np.concatenate([np.array([0.0]), beta_cml[0 : l - 1]])
        gamma_cml = np.cumsum(gammas) - gammas[0]
        alphas = beta_cml + gamma_cml

        delta_x = np.sum(a * np.cos(alphas) - b * np.sin(alphas))
        delta_y = np.sum(a * np.sin(alphas) + b * np.cos(alphas))

        if as_tuple:
            return delta_x, delta_y
        return np.sqrt(delta_x ** 2 + delta_y ** 2)
    
    # Get all module segment end points and module segment starting angles
    @staticmethod
    def get_mdl_seg_geo(sp, alpha, beta):
        angs = alpha + np.linspace(0, beta, GMRC.num_seg_lens)
        xy = np.zeros((GMRC.num_seg_lens + 1, 2))
        xy[:, 0] = xy[:, 0] + sp[0]
        xy[:, 1] = xy[:, 1] + sp[1]
        xy[1:, 0] = xy[1:, 0] + np.cumsum(np.cos(angs) * GMRC.mdl_seg_lens)
        xy[1:, 1] = xy[1:, 1] + np.cumsum(np.sin(angs) * GMRC.mdl_seg_lens)
        return (angs, xy)
    
    # Get module collider, includinh bounding box and body linestring
    @staticmethod
    def get_module_collider(geometry):
        angs, xy = GMRC.get_mdl_seg_geo(geometry[0], geometry[1], geometry[2])

        inner_pts = np.zeros((GMRC.num_seg_lens, 2))
        outer_pts = np.zeros((GMRC.num_seg_lens, 2))

        cos_ang_axis = np.cos(angs) * GMRC.plg_axs_place * GMRC.mdl_seg_lens
        sin_ang_axis = np.sin(angs) * GMRC.plg_axs_place * GMRC.mdl_seg_lens
        cos_ang_radius = np.cos(angs) * GMRC.radius_ratio
        sin_ang_radius = np.sin(angs) * GMRC.radius_ratio

        inner_pts[:, 0] = xy[:-1, 0] + cos_ang_axis + sin_ang_radius
        inner_pts[:, 1] = xy[:-1, 1] + sin_ang_axis - cos_ang_radius
        outer_pts[:, 0] = xy[:-1, 0] + cos_ang_axis - sin_ang_radius
        outer_pts[:, 1] = xy[:-1, 1] + sin_ang_axis + cos_ang_radius

        starting_line_seg = np.zeros((2, 2))
        starting_line_seg[0, 0] = xy[0, 0] + \
            np.cos(angs[0]) * GMRC.cls_exp_ratio * GMRC.mdl_seg_lens[0]
        starting_line_seg[0, 1] = xy[0, 1] + \
            np.sin(angs[0]) * GMRC.cls_exp_ratio * GMRC.mdl_seg_lens[0]
        starting_line_seg[1, 0] = xy[0, 0] + \
            np.cos(angs[0]) * GMRC.plg_axs_place[0] * GMRC.mdl_seg_lens[0]
        starting_line_seg[1, 1] = xy[0, 1] + \
            np.sin(angs[0]) * GMRC.plg_axs_place[0] * GMRC.mdl_seg_lens[0]

        ending_line_seg = np.zeros((2, 2))
        ending_line_seg[0, 0] = xy[-2, 0] + \
            np.cos(angs[-1]) * GMRC.plg_axs_place[-1] * GMRC.mdl_seg_lens[-1]
        ending_line_seg[0, 1] = xy[-2, 1] + \
            np.sin(angs[-1]) * GMRC.plg_axs_place[-1] * GMRC.mdl_seg_lens[-1]
        ending_line_seg[1, 0] = xy[-2, 0] + \
            np.cos(angs[-1]) * (1 - GMRC.cls_exp_ratio) * GMRC.mdl_seg_lens[-1]
        ending_line_seg[1, 1] = xy[-2, 1] + \
            np.sin(angs[-1]) * (1 - GMRC.cls_exp_ratio) * GMRC.mdl_seg_lens[-1]

        linestring_1 = np.vstack((starting_line_seg, outer_pts, ending_line_seg[0, :]))
        linestring_2 = np.vstack((starting_line_seg[1, :], inner_pts, ending_line_seg))

        mls = MultiLineString([linestring_1, linestring_2])
        bounds = mls.bounds
        return (box(bounds[0], bounds[1], bounds[2], bounds[3]), mls)
    
    # Get module collider for dangerous distance calculation
    # Return: (mls, hls, tls)
    # mls: MultiLineString for the whole module shape collider
    # hls: LineString for head shape collider
    # tls: LineString for tail shape collider
    @staticmethod
    def _get_dd_collider(geometry):
        angs, xy = GMRC.get_mdl_seg_geo(geometry[0], geometry[1], geometry[2])

        inner_pts = np.zeros((GMRC.num_seg_lens, 2))
        outer_pts = np.zeros((GMRC.num_seg_lens, 2))

        cos_ang_axis = np.cos(angs) * GMRC.plg_axs_place * GMRC.mdl_seg_lens
        sin_ang_axis = np.sin(angs) * GMRC.plg_axs_place * GMRC.mdl_seg_lens
        cos_ang_radius = np.cos(angs) * GMRC.radius_ratio
        sin_ang_radius = np.sin(angs) * GMRC.radius_ratio

        inner_pts[:, 0] = xy[:-1, 0] + cos_ang_axis + sin_ang_radius
        inner_pts[:, 1] = xy[:-1, 1] + sin_ang_axis - cos_ang_radius
        outer_pts[:, 0] = xy[:-1, 0] + cos_ang_axis - sin_ang_radius
        outer_pts[:, 1] = xy[:-1, 1] + sin_ang_axis + cos_ang_radius

        starting_line_seg = np.zeros((2, 2))
        starting_line_seg[0, 0] = xy[0, 0] + \
            np.cos(angs[0]) * GMRC.cls_exp_ratio * GMRC.mdl_seg_lens[0]
        starting_line_seg[0, 1] = xy[0, 1] + \
            np.sin(angs[0]) * GMRC.cls_exp_ratio * GMRC.mdl_seg_lens[0]
        starting_line_seg[1, 0] = xy[0, 0] + \
            np.cos(angs[0]) * GMRC.plg_axs_place[0] * GMRC.mdl_seg_lens[0]
        starting_line_seg[1, 1] = xy[0, 1] + \
            np.sin(angs[0]) * GMRC.plg_axs_place[0] * GMRC.mdl_seg_lens[0]

        ending_line_seg = np.zeros((2, 2))
        ending_line_seg[0, 0] = xy[-2, 0] + \
            np.cos(angs[-1]) * GMRC.plg_axs_place[-1] * GMRC.mdl_seg_lens[-1]
        ending_line_seg[0, 1] = xy[-2, 1] + \
            np.sin(angs[-1]) * GMRC.plg_axs_place[-1] * GMRC.mdl_seg_lens[-1]
        ending_line_seg[1, 0] = xy[-2, 0] + \
            np.cos(angs[-1]) * (1 - GMRC.cls_exp_ratio) * GMRC.mdl_seg_lens[-1]
        ending_line_seg[1, 1] = xy[-2, 1] + \
            np.sin(angs[-1]) * (1 - GMRC.cls_exp_ratio) * GMRC.mdl_seg_lens[-1]

        linestring_1 = np.vstack((starting_line_seg, outer_pts, ending_line_seg[0, :]))
        linestring_2 = np.vstack((starting_line_seg[1, :], inner_pts, ending_line_seg))
        mls = MultiLineString([linestring_1, linestring_2])

        start_linestring_1 = LineString(np.vstack((starting_line_seg, 
                                                   outer_pts, 
                                                    inner_pts[::-1, :], 
                                                    starting_line_seg[1, :])))
        end_linestring_2 = LineString(np.vstack((ending_line_seg[::-1, :], 
                                                    inner_pts[::-1, :], 
                                                    outer_pts, 
                                                    ending_line_seg[0, :])))
        
        return (mls, start_linestring_1, end_linestring_2)

    @staticmethod
    # Input parameters: 
    # axis, starting point, starting angle, arc radius
    # gripper 1 name, module name, gripper 2 name, number of points
    # simple: Whether to plot simplified node
    # h_node: If not None, the head gripper represented node
    # t_node: If not None, the tail gripper represented node
    def draw_module(ax, sp, sa, ba, g1n, mn, g2n, 
                    simple = False, h_node = None, t_node = None):
        angs, xy = GMRC.get_mdl_seg_geo(sp, sa, ba)
        ax.plot(xy[:, 0], xy[:, 1], '-k', alpha=0.5)

        texts = [g1n, mn, g2n]
        colors = ['moccasin', 'paleturquoise', 'moccasin']
        for i in range(len(texts)):
            if simple and i != 1:
                continue
            seg_i = GMRC.text_place[i][0]
            seg_r = GMRC.text_place[i][1]
            x = xy[seg_i, 0] + np.cos(angs[seg_i]) * seg_r * GMRC.mdl_seg_lens[i]
            y = xy[seg_i, 1] + np.sin(angs[seg_i]) * seg_r * GMRC.mdl_seg_lens[i]
            ax.text(x, y, texts[i], 
                    ha = 'center',
                    va = 'center', 
                    bbox=dict(boxstyle="round,pad=0.5", 
                              facecolor=colors[i], 
                              alpha=0.5))
            
        if simple and h_node is not None:
            ax.text(xy[0, 0], xy[0, 1], f"{h_node}", 
                    ha = 'center',
                    va = 'center', 
                    bbox=dict(boxstyle="round,pad=0.5", 
                              facecolor=colors[i], 
                              alpha=0.5))

        if simple and t_node is not None:
            ax.text(xy[-1, 0], xy[-1, 1], f"{t_node}", 
                    ha = 'center',
                    va = 'center', 
                    bbox=dict(boxstyle="round,pad=0.5", 
                              facecolor=colors[i], 
                              alpha=0.5))
            
    @staticmethod
    def get_random_configuration(m, seed = None, w = None, v = None):
        params = GMRC.get_random_configuration_model(m ,seed, w, v)
        return GMRC(*params)