"""
Continuum Modular Robot Reconfigure Planner: Task and Motion Planning
"""

from __future__ import annotations
import sys
sys.path.append('..')
sys.path.append('../GENN')
import copy
import warnings
import numpy as np
from GMRC import GMRC
from GENN import GENN, DegreeEmbedding, SequentialPooling

# Sometimes modifying grasping angle brings this complaint, just ignoring it here
warnings.filterwarnings(
    "ignore",
    message=r"^delta_grad == 0\.0\. Check if the approximated function is linear",
    category=UserWarning,
)

# The manager for correspondence and Gamma_final
class CGFManager:
    m: int = 7                              # Number of modules in a configuration

    # The initialization methods should only be called for the root node, once
    def __init__(self, Gamma_final: list, survival_idx: list = None, 
                 correspondence: list = None, num_corresponded: int = 0,
                 gf_idx_list: list = None, gt_idx_list: list = None):
        self.Gamma_final = Gamma_final
        
        if survival_idx is None: 
            self.survival_idx = [i for i in range(len(Gamma_final))]
        else:
            self.survival_idx = survival_idx

        # correspondence[2 * i + 0] = 2 * j + 1 => i's head is j's tail in target gmrc
        if correspondence is None:
            self.correspondence = [-1] * (2 * CGFManager.m)
        else:
            self.correspondence = correspondence

        # Number of modules that have been corresponded
        self.num_corresponded = num_corresponded

        # Avaialble Gamma_final indexes for a gripper as gf or gt
        if gf_idx_list is None or gt_idx_list is None: 
            gf_idx_list = [[]] * (2 * CGFManager.m)
            gt_idx_list = [[]] * (2 * CGFManager.m)
            for i in self.survival_idx:
                gamma_final = self.Gamma_final[i]
                gf = gamma_final[1]
                gt = gamma_final[2]
                gf_idx_list[gf].append(i)
                gt_idx_list[gt].append(i)
        self.gf_idx_list = gf_idx_list      # If chosen as gf, what angle can it choose
        self.gt_idx_list = gt_idx_list      # If chosen as gt, what angle can it choose

    # Collect an angle given Gamma_final index, and build correspondence based on it
    def get_angle(self, new_gf, new_gt, index):
        gamma_final = self.Gamma_final[index]
        angle = gamma_final[0]
        gf = gamma_final[1]
        gt = gamma_final[2]
        bias_index = gamma_final[3]
        is_w_grip = gamma_final[4]

        emt_new_gf = 2 * (new_gf // 2) + 1 - new_gf % 2
        emt_gf = 2 * (gf // 2) + 1 - gf % 2
        emt_new_gt = 2 * (new_gt // 2) + 1 - new_gt % 2
        emt_gt = 2 * (gt // 2) + 1 - gt % 2

        if self.correspondence[new_gf] < 0:
            self.correspondence[new_gf] = gf
            self.correspondence[emt_new_gf] = emt_gf
            self.num_corresponded = self.num_corresponded + 1
        if self.correspondence[new_gt] < 0:
            self.correspondence[new_gt] = gt
            self.correspondence[emt_new_gt] = emt_gt
            self.num_corresponded = self.num_corresponded + 1
        
        if not is_w_grip:
            new_survival_idx = []
            for idx in self.survival_idx:
                if not (idx >= index - bias_index and idx < index - bias_index + 2):
                    new_survival_idx.append(idx)
            self.survival_idx = new_survival_idx
            self.gf_idx_list[gf] = []
            self.gf_idx_list[gt] = []
            self.gt_idx_list[gt] = []
            self.gt_idx_list[gf] = []
        else:
            keep_idx = (index - bias_index) + (bias_index + 2) % 6
            new_survival_idx = []
            for idx in self.survival_idx:
                if not (idx >= index - bias_index and idx < index - bias_index + 6):
                    new_survival_idx.append(idx)
                elif idx == keep_idx:
                    new_survival_idx.append(idx)
            self.survival_idx = new_survival_idx
            self.gf_idx_list[gf] = []           # gf can no longer grasp any gripper
            self.gf_idx_list[gt] = []           # gt can no longer grasp any gripper
            self.gt_idx_list[gt] = []           # gt can no longer be grasper
            self.gt_idx_list[gf] = [keep_idx]   # gf can only be grasped with keep_idx
            
        return angle

    # Get the appropriate Gamma_final angle given grippers in new gmrc (not in target)
    #   Each returned index collects a angle from cgf_manager to build correspondence
    def get_angle_idxes(self, new_gf, new_gt, is_w_grip):
        gf = self.correspondence[new_gf]
        gt = self.correspondence[new_gt]
        if gt < 0 and is_w_grip:            # If grasp a uncorresponded outer gripper
            return []                       # Only can build correspondence bottom->top
        elif gf < 0 and gt < 0:             # If both grippers are uncorresponded
            return self.survival_idx
        elif gf >= 0 and gt < 0:            # If only gripper_from is corresponded
            return self.gf_idx_list[gf]
        elif gf < 0 and gt >= 0:            # If only gripper_to is corresponded
            return self.gt_idx_list[gt]
        else:                               # If both grippers are corresponded
            idx_list = []
            for idx in self.gf_idx_list[gf]:
                if self.Gamma_final[idx][2] == gt:
                    idx_list.append(idx)
            return idx_list

    def copy(self):                         # Corresponding will not change ang_lists
        return CGFManager(Gamma_final=self.Gamma_final,
                          survival_idx=copy.deepcopy(self.survival_idx),
                          correspondence=copy.deepcopy(self.correspondence),
                          num_corresponded=self.num_corresponded,
                          gf_idx_list=copy.deepcopy(self.gf_idx_list),
                          gt_idx_list=copy.deepcopy(self.gt_idx_list))

# A search tree node contains: 1. a unique gmrc shape; 2. a partial correspondence
class TreeNode:
    mediocrity_tolerance = 3

    def __init__(self, gmrc: GMRC, cgf_manager: CGFManager, 
                 parent: TreeNode = None, g_depth: int = 0,
                 mediocrity: int = 0, tree: Tree = None):
        self.cgf_manager = cgf_manager
        self.gmrc: GMRC = gmrc
        self.parent: TreeNode = parent
        self.g_depth: int = g_depth
        self.children: list[TreeNode] = []
        self.mediocrity: int = mediocrity
        self.tree: Tree = tree
        self.expanded = False

        self.tree.add_node_to_depth(self, self.g_depth)
        
    def expand_to(self, tar_g_depth = None):
        if tar_g_depth is not None:
            if self.g_depth >= tar_g_depth:
                return
        
        if not self.expanded:
            actions = self.gmrc.get_all_actions()
            for action in actions:
                new_gmrc = self.gmrc.copy()
                if not isinstance(action, tuple):               # Release
                    new_gmrc.execute_action(action)
                    new_node = TreeNode(new_gmrc, self.cgf_manager.copy(),
                                        self, self.g_depth, self.mediocrity, self.tree)
                    self.children.append(new_node)
                else:                                           # Grasp
                    if not new_gmrc.execute_action(action):
                        continue
                    self.children.extend(
                        self._get_all_memebers_in_grasping_group(new_gmrc, action))
        self.expanded = True

        self.children.reverse()
        for child in self.children:
            if child.is_goal():
                return child
            grand_child = child.expand_to(tar_g_depth)
            if grand_child is not None:
                return grand_child

        return None

    def expand(self, extra_depth: int = 1):
        return self.expand_to(self.g_depth + extra_depth)
    
    # Whether this node is the goal configuration
    def is_goal(self):
        if len(self.cgf_manager.survival_idx) == 0:
            return True
        else:
            return False

    # Get all reasonable children from the same grasping action based on eldest sibling
    # NOTE: Will not have a->b with alpha and b->a with alpha
    #       since will only have a->b from EMRC.get_all_actions()
    def _get_all_memebers_in_grasping_group(self, new_gmrc: GMRC, action: tuple):
        if self.mediocrity < TreeNode.mediocrity_tolerance:
            optim_node = TreeNode(new_gmrc, self.cgf_manager.copy(),
                                self, self.g_depth + 1, self.mediocrity + 1, self.tree)
            members = [optim_node]
        else:
            members = []
        if len(new_gmrc.module_loops[-1]) <= 2:
            return members
        gf = action[0]
        gt = action[1]
        grip = new_gmrc.module2gripper[gf % 2][gf // 2] // 3
        mid_ang = new_gmrc.get_grip_gamma(grip)

        min_gmrc = new_gmrc.copy()
        min_gmrc.modify_grsp_ang(grip, -GMRC.grsp_ang_cap)
        min_ang = min_gmrc.get_grip_gamma(grip)
        if min_ang < mid_ang and self.mediocrity < TreeNode.mediocrity_tolerance:
            min_node = TreeNode(min_gmrc, self.cgf_manager.copy(),
                                self, self.g_depth + 1, self.mediocrity + 1, self.tree)
            members.append(min_node)

        max_gmrc = new_gmrc.copy()
        max_gmrc.modify_grsp_ang(grip, GMRC.grsp_ang_cap)
        max_ang = max_gmrc.get_grip_gamma(grip)
        if max_ang > mid_ang and self.mediocrity < TreeNode.mediocrity_tolerance:
            max_node = TreeNode(max_gmrc, self.cgf_manager.copy(),
                                self, self.g_depth + 1, self.mediocrity + 1, self.tree)
            members.append(max_node)

        is_w_grip = new_gmrc.is_grip_w[grip]
        all_Gamma_final_idxs = self.cgf_manager.get_angle_idxes(gf, gt, is_w_grip)
        for idx in all_Gamma_final_idxs:
            # "bc" means building correspondence here
            bc_cgf_manager = self.cgf_manager.copy()
            ang = bc_cgf_manager.get_angle(gf, gt, idx)
            if ang <= min_ang and ang >= max_ang:
                continue
            bc_gmrc = new_gmrc.copy()
            bc_gmrc.modify_grsp_ang(grip, ang)
            if np.abs(bc_gmrc.get_grip_gamma(grip) - ang) > 1e-3:
                continue
            bc_node = TreeNode(bc_gmrc, bc_cgf_manager, 
                               self, self.g_depth + 1, 0, self.tree)
            members.append(bc_node)
            if bc_node.is_goal():
                break

        return members

class Tree:
    def __init__(self, gmrc: GMRC, cgf_manager: CGFManager):
        self.nodes_at_depth: list[TreeNode] = [[]]
        self.max_g_depth = 0
        self.root = TreeNode(gmrc,
                        cgf_manager=cgf_manager,
                        parent=None,
                        g_depth=0,
                        mediocrity=0,
                        tree = self)

    def add_node_to_depth(self, node: TreeNode, g_depth: int):
        while g_depth > self.max_g_depth:
            self.nodes_at_depth.append([])
            self.max_g_depth = self.max_g_depth + 1
        self.nodes_at_depth[g_depth].append(node)

    def push_front(self):
        goal_node = None
        max_g_depth_before = self.max_g_depth
        for node in self.nodes_at_depth[-1]:
            goal_node = node.expand()
            if goal_node is not None:
                break
        max_g_depth_after = self.max_g_depth
        if max_g_depth_before == max_g_depth_after:
            raise RuntimeError("Can not further expand any leaf nodes!")
        num_new_nodes = len(self.nodes_at_depth[-1])
        print(f"Find {num_new_nodes} nodes at depth {len(self.nodes_at_depth) - 1}")
        return goal_node

class CMRRP:
    def __init__(self, g: GENN = None,
                 d: DegreeEmbedding = None,
                 s: SequentialPooling = None):
        self.genn = g
        self.degree_embedding = d
        self.sequential_pooling = s

    def plan(self, gmrc_1: GMRC, gmrc_2: GMRC):
        GMRC.suppress_action_err = True
        assert gmrc_1.m == gmrc_2.m
        CGFManager.m = gmrc_1.m
        cgf_manager = CGFManager(gmrc_2.get_Gamma_final())
        tree = Tree(gmrc_1, cgf_manager)
        while True:
            node = tree.push_front()
            if node is not None:
                path = [node]
                while node.parent is not None:
                    node = node.parent
                    path.append(node)
                path.reverse()
                return path