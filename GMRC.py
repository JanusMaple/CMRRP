import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from shapely import STRtree
from shapely.geometry import MultiLineString, box

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
        self.wgpa_obj = []      # w-grip angle objective (360 for cc and 720 for c)
        # All these lists have elements of int arrays, serving as indexes for np.array
        self.ba_xi_loops = []   # A list of lists of indexes of x for bend_angs
        self.ga_xi_loops = []   # A list of lists of indexes of x for grsp_angs
        # All these lists have elements of np.array for accelerating optimizing
        self.bas_loops = []     # The bend angle sign for each bend angle in loop
        self.gas_loops = []     # The grasp angle sign for each grasp angle in loop

        if bending_angles is None or grasping_angles is None:
            is_planar, embedding = nx.check_planarity(self.G)   # Must be planar
            cannot_be_docked = False
            for try_collision_free in range(20):
                self.bend_angs, self.grsp_angs = self.get_random_angles_da()
                if cannot_be_docked or not is_planar:
                    break
                if len(self.module_loops) > 0:      # If the graph is not acyclic
                    error = 1
                    for try_docking_loops in range(3):
                        x0 = self.initialize_x()    # Initialize all variables
                        x0 = self.optim_angles_la(x0)
                        x = self.optim_angles_ld(x0)
                        error = self.get_loop_dock_error(x)
                        if error > 1e-3:
                            self.bend_angs, self.grsp_angs = self.get_random_angles_da()
                        else:
                            break                   # Until global minimized
                    else:
                        cannot_be_docked = True
                    self.update_angs_from_x(x)
                # Format: {module_index: ((x, y), alpha, beta), ...}
                self.module_geometries = {}         # The position, orientation of arcs
                # Format: [(Bounding Box, Body Polygon), ...]
                self.module_colliders = []          # The bounding box and body polygon
                self.update_all_module_geometry()   # Update module_geometries
                if not self.is_collision_detected():
                    break                           # Until no collision
            else:
                if cannot_be_docked:
                    print('\033[91mFailed for Docking Loops :(\033[0m')
                elif not is_planar:
                    print('\033[91mThe Graph is Not Planar :(\033[0m')
                else:
                    print('\033[91mFailed for Finding Collision-Free Layout :(\033[0m')
        else:
            self.bend_angs = bending_angles
            self.grsp_angs = grasping_angles
            # Format: {module_index: ((x, y), alpha, beta)}
            self.module_geometries = {}         # The position, orientation of arcs
            # Format: [(Bounding Box, Body Polygon), ...]
            self.module_colliders = []          # The bounding box and body polygon
            self.update_all_module_geometry()   # Update module_geometries

    # Generate random angles under dock-angle constraint
    def get_random_angles_da(self):
        bend_angs = self.rng.uniform(-GMRC.mdl_ang_cap, GMRC.mdl_ang_cap, self.m)
        grsp_angs = np.zeros(len(self.grippers))
        for grip in range(self.w + self.v):
            if self.is_grip_w[grip]:
                grsp_angs[3 * grip : 3 * (grip + 1)] = self.get_random_grip_angles(grip)
            else:
                grsp_angs[3 * grip] = self.rng.uniform(
                    -GMRC.grsp_ang_cap, GMRC.grsp_ang_cap)
        return bend_angs, grsp_angs
    
    # Initialize x from angles (self.bend_angs and self.grsp_angs)
    def initialize_x(self):
        nmil = 0
        mi2xi = [-1] * self.m
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
    
    # Optimize angles to minimize loop-angle and w_grip_angle error
    def optim_angles_la(self, x0):
        if len(self.xi_wf) > 0:
            constraints = [
                {
                    'type': 'eq', 
                    'fun': self.get_w_grip_angle_error
                }
            ]
        else:
            constraints = []
        result = minimize(
            self.get_loop_angle_error, 
            x0, 
            method='SLSQP', 
            constraints=constraints, 
            bounds=self.x_boundary)
        return result.x
    
    # Optimize angles to minimize loop-dock error
    def optim_angles_ld(self, x0):
        if len(self.xi_wf) > 0:
            constraints = [
                {
                    'type': 'eq', 
                    'fun': self.get_w_grip_angle_error
                }, 
                {
                    'type': 'eq', 
                    'fun': self.get_loop_angle_error
                }
            ]
        else:
            constraints = [
                {
                    'type': 'eq', 
                    'fun': self.get_loop_angle_error
                }
            ]
        result = minimize(
            self.get_loop_dock_error, 
            x0, 
            method='SLSQP', 
            constraints=constraints, 
            bounds=self.x_boundary)
        return result.x
    
    # Get w-grip angle error (from 360 or 720 depending on the direction)
    def get_w_grip_angle_error(self, x):
        error = 0.0
        for i in range(len(self.xi_wf)):
            j = self.xi_wf[i]
            error = error + np.abs(x[j] + x[j + 1] + x[j + 2] - self.wgpa_obj[i])
        return error

    # Get loop-angle error for x
    def get_loop_angle_error(self, x):
        error = 0
        for i in range(len(self.ba_xi_loops)):
            error = error + np.abs(
                np.sum(x[self.ba_xi_loops[i]] * self.bas_loops[i])
                + np.sum(x[self.ga_xi_loops[i]] * self.gas_loops[i])
                - self.loop_polarities[i] * 2 * np.pi)
        return error

    # Get loop-dock error for x
    def get_loop_dock_error(self, x, is_print = False):
        error = 0
        for i in range(len(self.ba_xi_loops)):
            betas = x[self.ba_xi_loops[i]] * self.bas_loops[i]
            gammas = x[self.ga_xi_loops[i]] * self.gas_loops[i]

            l = len(self.ba_xi_loops[i])
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
            error = error + np.sqrt(delta_x ** 2 + delta_y ** 2)
            if is_print:
                print('*********************')
                print("Loop id: ", end='')
                print(i)
                print("Module angles: ", end='')
                print(betas)
                print("Grasp angles: ", end='')
                print(gammas)
                print("Starting angles: ", end='')
                print(alphas)
                print('Module a: ', end='')
                print(a)
                print('Module b: ', end='')
                print(b)
                print('Loop docking error: ', end='')
                print((delta_x, delta_y))
        if is_print:
            print('*********************')
            print('Total docking error: ', end='')
            print(error)
        return error
    
    # Update angles (self.bend_angs and self.grsp_angs) from x
    def update_angs_from_x(self, x):
        for i in range(self.number_module_in_loop):
            self.bend_angs[self.xi2angi[i]] = x[i]
        for i in range(self.number_module_in_loop, len(x)):
            self.grsp_angs[self.xi2angi[i]] = x[i]

    # Generate random angles for a w-grip with polarity
    def get_random_grip_angles(self, grip):
        # Range of the sum of gamma1 and gamma2
        if self.grip_polarities[grip] == 1:
            gamma_sum_min = -np.pi - GMRC.grsp_ang_cap
            gamma_sum_max = -np.pi + GMRC.grsp_ang_cap
        elif self.grip_polarities[grip] == -1:
            gamma_sum_min = np.pi - GMRC.grsp_ang_cap
            gamma_sum_max = np.pi + GMRC.grsp_ang_cap

        gamma1_min = max(gamma_sum_min - GMRC.grsp_ang_cap, -GMRC.grsp_ang_cap)
        gamma1_max = min(gamma_sum_max + GMRC.grsp_ang_cap, GMRC.grsp_ang_cap)
        gamma1 = self.rng.uniform(gamma1_min, gamma1_max)

        gamma2_min = max(gamma_sum_min - gamma1, -GMRC.grsp_ang_cap)
        gamma2_max = min(gamma_sum_max - gamma1, GMRC.grsp_ang_cap)
        gamma2 = self.rng.uniform(gamma2_min, gamma2_max)
        
        if self.grip_polarities[grip] == 1:
            gamma3 = 2 * np.pi - (np.pi * 3) - gamma1 - gamma2  # 360 Constraint
        elif self.grip_polarities[grip] == -1:
            gamma3 = 4 * np.pi - (np.pi * 3) - gamma1 - gamma2  # 720 Constraint

        return [gamma1, gamma2, gamma3]

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
            
    # Update geometries and colliders of all modules
    def update_all_module_geometry(self):
        self.mdl_geo_updated = [False] * self.m
        self.module_geometries = {}
        self.update_module_geometry(0, (0, 0), 0, 0)

        self.module_colliders = []
        for i in range(self.m):
            self.module_colliders.append(
                self.get_module_collider(self.module_geometries[i]))

    # Update geometry recursively
    def update_module_geometry(self, mi, sp, sa, ht):
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
            self.bend_angs[mi]
            )
        self.mdl_geo_updated[mi] = True

        for i in range(2):
            if self.module2gripper[i][mi] >= 0: # If it participates in a grip
                gripper = self.module2gripper[i][mi]
                gns = self.get_gripper_grip_neighbors(gripper)  # Gripper Grip Neighbors
                for gn in gns:
                    mn = self.gripper2module[gn] // 2           # Module Neighbor
                    mn_ht = self.gripper2module[gn] % 2
                    if not self.mdl_geo_updated[mn]:
                        mn_sp = ht_position[i]
                        mn_sa = ht_angle[i] + self.get_grasp_angle(gripper, gn)
                        self.update_module_geometry(mn, mn_sp, mn_sa, mn_ht)

    # Get module collider, includinh bounding box and body linestring
    def get_module_collider(self, geometry): 
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

    # To update: bend_angs, grsp_angs, module_geometries, module_colliders
    def execute_action(self, action):
        cyc_status, grip_status = super().execute_action(action)
        if isinstance(action, tuple):
            GMRC._execute_grasping(self, cyc_status, grip_status)
        else:
            GMRC._execute_releasing(self, cyc_status, grip_status)

    def _execute_grasping(self, cyc_status, grip_status):
        pass

    def _execute_releasing(self, cyc_status, grip_status):
        pass

    def print_all_angs(self):
        np.set_printoptions(precision=2, suppress=True, linewidth=1024)
        print(f"Bending angles are {self.bend_angs}")
        print(f"Grasping angles are {self.grsp_angs}")

    def show_all(self):
        self.print_all()
        self.show_topology()
        self.show_geometry()

    def show_geometry(self):
        
        if self.module_geometries == {}:
            return

        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_aspect('equal')
        ax.axis('off')

        for i in range(self.m):
            g1n = f"H{self.module2gripper[0][i]}"
            mn = f"{i}"
            g2n = f"T{self.module2gripper[1][i]}"
            GMRC.draw_module(
                ax, 
                self.module_geometries[i][0], 
                self.module_geometries[i][1], 
                self.module_geometries[i][2], 
                g1n, 
                mn, 
                g2n
            )
            for line in self.module_colliders[i][1].geoms:
                x, y = line.xy
                ax.plot(x, y, color = 'b')

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
        print("-----------------------------------------------------------------")
        self.print_actions()
        print("-----------------------------------------------------------------")

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

    @staticmethod
    # Input parameters: 
    # axis, starting point, starting angle, arc radius
    # gripper 1 name, module name, gripper 2 name, number of points
    def draw_module(ax, sp, sa, ba, g1n, mn, g2n):
        angs, xy = GMRC.get_mdl_seg_geo(sp, sa, ba)
        ax.plot(xy[:, 0], xy[:, 1], '-k')

        texts = [g1n, mn, g2n]
        colors = ['red', 'blue', 'red']
        for i in range(len(texts)):
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
            
    @staticmethod
    def get_random_configuration(m, seed = None, w = None, v = None):
        params = GMRC.get_random_configuration_model(m ,seed, w, v)
        return GMRC(*params)