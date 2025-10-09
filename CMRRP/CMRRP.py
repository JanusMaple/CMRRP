"""
Continuum Modular Robot Reconfigure Planner: Task and Motion Planning
"""

from __future__ import annotations
import sys
sys.path.append('..')
sys.path.append('../GENN')
sys.path.append('../GGNN')
import os, multiprocessing as mp
import copy
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from itertools import repeat
import numpy as np
import torch
import torch.nn.functional as F
from GMRC import GMRC
from EMRC import EMRC
import GENN, GGNN

# Sometimes modifying grasping angle brings this complaint, just ignoring it here
warnings.filterwarnings(
    "ignore",
    message=r"^delta_grad == 0\.0\. Check if the approximated function is linear",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message="Singular Jacobian matrix. Using SVD decomposition"
)

# The manager for correspondence and Gamma_final
class CGFManager:
    m: int = 7                              # Number of modules in a configuration

    # The initialization methods should only be called for the root node, once
    def __init__(self, Gamma_final: list, survival_idx: list = None, 
                 correspondence: list = None, num_constructed: int = 0,
                 gf_idx_list: list = None, gt_idx_list: list = None, 
                 can_release = None, is_cursed = False, curse = None,
                 built_grip_parity = 0):
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

        # Number of constructive grasps that have been taken
        self.num_constructed = num_constructed

        # Whether a gripper can release (False if has participated in a grasp)
        if can_release is None:
            self.can_release = [True] * (2 * CGFManager.m)
        else:
            self.can_release = can_release

        # Whether the current configuration is cursed and has to fulfill the curse
        self.is_cursed = is_cursed
        self.curse = curse

        # Avaialble Gamma_final indexes for a gripper as gf or gt
        if gf_idx_list is None or gt_idx_list is None: 
            gf_idx_list = [[] for _ in range(2 * CGFManager.m)]
            gt_idx_list = [[] for _ in range(2 * CGFManager.m)]
            for i in self.survival_idx:
                gamma_final = self.Gamma_final[i]
                gf = gamma_final[1]
                gt = gamma_final[2]
                gf_idx_list[gf].append(i)
                gt_idx_list[gt].append(i)
        self.gf_idx_list = gf_idx_list      # If chosen as gf, what angle can it choose
        self.gt_idx_list = gt_idx_list      # If chosen as gt, what angle can it choose

        # A hash value for documenting which grips have been reconstructed
        self.built_grip_parity = built_grip_parity

    # Collect an angle given Gamma_final index, and build correspondence based on it
    def get_angle(self, new_gf, new_gt, index):
        self.can_release[new_gf] = False
        self.can_release[new_gt] = False

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
        if self.correspondence[new_gt] < 0:
            self.correspondence[new_gt] = gt
            self.correspondence[emt_new_gt] = emt_gt
        
        self.num_constructed = self.num_constructed + 1
        
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
            grip = self.Gamma_final[index][5]
            self.built_grip_parity = self.built_grip_parity + pow(2, grip)
        else:
            # For w-grip in Gamma_final:
            #   [..., gamma_1, -gamma_1, gamma_2, -gamma_2, gamma_3, -gamma_3, ...]
            if bias_index % 2 == 0:
                # gamma_1 => gamma_2; gamma_2 => gamma_3; gamma_3 => gamma_1
                keep_idx = (index - bias_index) + (bias_index + 2) % 6
            else:
                # -gamma_1 => -gamma_3; -gamma_2 => -gamma_1; -gamma_3 => -gamma_2
                keep_idx = (index - bias_index) + (bias_index + 4) % 6
            new_survival_idx = []
            w_grip_constructed = True
            for idx in self.survival_idx:
                if not (idx >= index - bias_index and idx < index - bias_index + 6):
                    new_survival_idx.append(idx)
                elif idx == keep_idx:
                    w_grip_constructed = False          # Survivor indicates uncompleted
                    new_survival_idx.append(idx)
            self.survival_idx = new_survival_idx
            if not w_grip_constructed:
                for gt_idxes in self.gt_idx_list:
                    for i in range(len(gt_idxes)):
                        if gt_idxes[i] == keep_idx:
                            gt_idxes[i : i + 1] = []    # Purge keep_idx first
                            break
                self.gf_idx_list[gf] = []           # gf can no longer grasp any gripper
                self.gf_idx_list[gt] = []           # gt can no longer grasp any gripper
                self.gt_idx_list[gt] = []           # gt can no longer be grasped
                self.gt_idx_list[gf] = [keep_idx]   # gf can be grasped with keep_idx
            else:
                self.gf_idx_list[gf] = []           # gf can no longer grasp any gripper
                self.gf_idx_list[gt] = []           # gt can no longer grasp any gripper
                self.gt_idx_list[gt] = []           # gt can no longer be grasped
                self.gt_idx_list[gf] = []           # gf can no longer be grasped
                grip = self.Gamma_final[index][5]
                self.built_grip_parity = self.built_grip_parity + pow(2, grip)
            
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
        
    # Get all non-trivial angle choices based on Gamma_final from target gmrc
    #   Including angles directly leading to correspondence and angles as stepping-stone
    """
    new_gf:     The gripper that is grasping another gripper
    new_gt:     The gripper that is going to be grasped by new_gf
    is_w_grip:  Whether the new formed grip will be a w-grip
    new_gi:     The inner gripper of the new formed w-grip if is forming a w-grip
    gamma_0:    The angle from new_gi to new_gt
    """
    def get_ang_choices(self, new_gf, new_gt, is_w_grip, new_gi = None, gamma_0 = None):
        choices = []
        choices.extend(self.get_angle_idxes(new_gf, new_gt, is_w_grip))
        cursed_choices = {}                 # Choices that are 2-loop stepping-stone

        emt_new_gf = 2 * (new_gf // 2) + 1 - new_gf % 2
        emt_new_gt = 2 * (new_gt // 2) + 1 - new_gt % 2
        idxes = self.get_angle_idxes(emt_new_gf, emt_new_gt, False)
        for idx in idxes:
            gamma = -self.Gamma_final[idx][0]
            built_grip = self.Gamma_final[idx][5]
            key = (gamma, built_grip)
            if not key in cursed_choices:
                cursed_choices[key] = []
            cursed_choices[key].append((emt_new_gf, emt_new_gt, idx, new_gf))
        idxes = self.get_angle_idxes(emt_new_gt, emt_new_gf, False)
        for idx in idxes:
            gamma = self.Gamma_final[idx][0]
            built_grip = self.Gamma_final[idx][5]
            key = (gamma, built_grip)
            if not key in cursed_choices:
                cursed_choices[key] = []
            cursed_choices[key].append((emt_new_gt, emt_new_gf, idx, new_gf))
        if is_w_grip:
            emt_new_gi = 2 * (new_gi // 2) + 1 - new_gi % 2
            idxes = self.get_angle_idxes(emt_new_gf, emt_new_gi, False)
            for idx in idxes:
                if self.Gamma_final[idx][0] < 0:
                    gamma = -np.pi - gamma_0 - self.Gamma_final[idx][0]
                else:
                    gamma = np.pi - gamma_0 - self.Gamma_final[idx][0]
                built_grip = self.Gamma_final[idx][5]
                key = (gamma, built_grip)
                if not key in cursed_choices:
                    cursed_choices[key] = []
                cursed_choices[key].append((emt_new_gf, emt_new_gi, idx, new_gf))
            idxes = self.get_angle_idxes(emt_new_gi, emt_new_gf, False)
            for idx in idxes:
                if self.Gamma_final[idx][0] < 0:
                    gamma = np.pi - gamma_0 - (-self.Gamma_final[idx][0])
                else:
                    gamma = -np.pi - gamma_0 - (-self.Gamma_final[idx][0])
                built_grip = self.Gamma_final[idx][5]
                key = (gamma, built_grip)
                if not key in cursed_choices:
                    cursed_choices[key] = []
                cursed_choices[key].append((emt_new_gi, emt_new_gf, idx, new_gf))

        choices.extend(list(cursed_choices.items()))
        return choices
    
    """
    curse: list[tuple, tuple, ..., tuple(gf, gt, idx, locked_gripper)]
    """
    def cursed_by(self, curse: list):
        self.is_cursed = True
        self.curse = curse
        self.can_release[self.curse[0][3]] = False

    def break_curse(self):
        self.is_cursed = False
        self.can_release[self.curse[0][3]] = True
        self.curse = None

    def copy(self):                         # Corresponding will not change ang_lists
        return CGFManager(Gamma_final=self.Gamma_final,
                          survival_idx=copy.deepcopy(self.survival_idx),
                          correspondence=copy.deepcopy(self.correspondence),
                          num_constructed=self.num_constructed,
                          gf_idx_list=copy.deepcopy(self.gf_idx_list),
                          gt_idx_list=copy.deepcopy(self.gt_idx_list),
                          can_release=copy.deepcopy(self.can_release),
                          is_cursed=self.is_cursed,
                          curse=copy.deepcopy(self.curse),
                          built_grip_parity=self.built_grip_parity)

# Parallel Optimizer for Grasping/Modifying of GMRC
class ParOptimizer:
    pool = None

    """
    Dock the gmrc by action naturally, and then find min/max grasp angle
    NOTE: Inputs and outputs are automatically pickled and copied
    Return: (natural_gmrc, (min_gmrc, min_ang), (max_gmrc, max_ang))
    """
    @staticmethod
    def dock_min_max(gmrc: GMRC, action: tuple):
        if not gmrc.execute_action(action):
            return (None, (None, None), (None, None))
        
        mid_gmrc = gmrc
        gf = action[0]
        grip = mid_gmrc.module2gripper[gf % 2][gf // 2] // 3
        mid_ang = mid_gmrc.get_grip_gamma(grip)

        if gmrc.is_2_cycle(-1):
            return (gmrc, (None, mid_ang), (None, mid_ang))
        
        min_gmrc = gmrc.copy()
        if not min_gmrc.modify_grsp_ang(grip, -GMRC.grsp_ang_cap):
            min_ang = mid_ang
            min_gmrc = None
        else:
            min_ang = min_gmrc.get_grip_gamma(grip)
            if min_ang >= mid_ang:
                min_gmrc = None

        max_gmrc = gmrc.copy()
        if not max_gmrc.modify_grsp_ang(grip, GMRC.grsp_ang_cap):
            max_ang = mid_ang
            max_gmrc = None
        else:
            max_ang = max_gmrc.get_grip_gamma(grip)
            if max_ang <= mid_ang:
                max_gmrc = None

        return (mid_gmrc, (min_gmrc, min_ang), (max_gmrc, max_ang))
    
    """
    Modify the angles of GMRCs and return the results
    NOTE: Inputs and outputs are automatically pickled and copied
    Returen: [modified_gmrc, ..., modified_gmrc]
    """
    @staticmethod
    def modify(gmrcs: list[GMRC], grips: list[int], angs: list[float]):
        modified_gmrcs = []
        for ori_gmrc, grip, ang in zip(gmrcs, grips, angs):
            gmrc = ori_gmrc.copy()
            if not gmrc.modify_grsp_ang(grip, ang):
                modified_gmrcs.append(None)
                continue
            modified_ang = gmrc.get_grip_gamma(grip)
            if np.abs(modified_ang - ang) > 0.5 / 180 * np.pi:
                modified_gmrcs.append(None)
                continue
            modified_gmrcs.append(gmrc)
        return modified_gmrcs

    @staticmethod
    def par_dock_min_max(gmrcs: list[GMRC], actions: list[tuple], max_workers: int):
        ParOptimizer.ensure_pool(max_workers)
        try:
            gmrcs = ParOptimizer.pool.map(ParOptimizer.dock_min_max, gmrcs, actions)
            return gmrcs
        except KeyboardInterrupt:
            ParOptimizer.reset_pool()
            raise
        except BrokenProcessPool:
            ParOptimizer.reset_pool()
            raise

    @staticmethod
    def par_modify(gmrcs: list[GMRC], grips: list[int],
                   angs: list[float], max_workers: int):
        ParOptimizer.ensure_pool(max_workers)
        try:
            num_atomic_tasks = len(angs)
            chunk_size = num_atomic_tasks // max_workers
            if chunk_size * max_workers < num_atomic_tasks:
                chunk_size = chunk_size + 1
            gmrc_chunks = []
            grip_chunks = []
            ang_chunks = []
            num_assigned = 0
            while num_assigned < num_atomic_tasks:
                gmrc_chunks.append(gmrcs[num_assigned : num_assigned + chunk_size])
                grip_chunks.append(grips[num_assigned : num_assigned + chunk_size])
                ang_chunks.append(angs[num_assigned : num_assigned + chunk_size])
                num_assigned = num_assigned + chunk_size
            modified_gmrc_chunks = ParOptimizer.pool.map(
                ParOptimizer.modify, gmrc_chunks, grip_chunks, ang_chunks)
            modified_gmrcs = []
            for modified_gmrc_chunk in modified_gmrc_chunks:
                modified_gmrcs.extend(modified_gmrc_chunk)
            return modified_gmrcs
        except KeyboardInterrupt:
            ParOptimizer.reset_pool()
            raise
        except BrokenProcessPool:
            ParOptimizer.reset_pool()
            raise

    @staticmethod
    def _init_worker():
        GMRC.suppress_action_err = True
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

    @staticmethod
    def make_pool(max_workers):
        return ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=mp.get_context("spawn"),
            initializer=ParOptimizer._init_worker
        )

    @staticmethod
    def ensure_pool(max_workers):
        if ParOptimizer.pool is None:
            ParOptimizer.pool = ParOptimizer.make_pool(max_workers)

    @staticmethod
    def reset_pool():
        if ParOptimizer.pool is not None:
            try:
                ParOptimizer.pool.shutdown(wait=False, cancel_futures=True)
            except Exception:
                pass
        ParOptimizer.pool = None

    @staticmethod
    def cleanup_all_pools() -> None:
        if ParOptimizer.pool is not None:
            try:
                ParOptimizer.pool.shutdown(wait=True, cancel_futures=True)
            except Exception:
                try:
                    ParOptimizer.pool.shutdown(wait=False, cancel_futures=True)
                except Exception:
                    pass
            finally:
                ParOptimizer.pool = None

# A search tree node contains: 1. a unique gmrc shape; 2. a partial correspondence
class TreeNode:
    mediocrity_tolerance = 0
    is_grouping = False
    max_workers = 12

    def __init__(self, gmrc: GMRC, cgf_manager: CGFManager, 
                 parent: TreeNode = None, g_depth: int = 0,
                 mediocrity: int = 0, tree: Tree = None, 
                 group_feature: tuple = None):
        self.cgf_manager = cgf_manager      # Partial Correspondence
        self.gmrc: GMRC = gmrc              # GMRC

        self.parent: TreeNode = parent      # Parent Tree Node

        self.g_depth: int = g_depth         # Grasping Depth

        self.children: list[TreeNode] = []  # Children Nodes

        self.mediocrity: int = mediocrity   # Current Mediocrity

        self.tree: Tree = tree              # Belonging Tree

        self.group_feature = group_feature  # Group feature for MCTS with group nodes
        
        self.actions = None                 # All potential actions

        self.release_expanded = False       # Whether have expanded releasing children
        self.expanded = False               # Whether have expanded all children

        self.id_ethnicity = None            # Tuple of node id and ethnicity

        self.is_novel = self.tree.add_node_to_depth(self)

    # Get both node identifier and its ethnicity (based on grsp_ang parity)
    def get_id_ethnicity(self):
        if self.id_ethnicity is None:
            self.id_ethnicity = self.tree.id_verdict.get_id_ethnicity(
                self.gmrc, self.cgf_manager)
        return self.id_ethnicity
        
    def expand_to(self, tar_depth = None, is_g_depth = True, cur_depth = 0):
        if tar_depth is not None:
            if is_g_depth:                                      # Expand to grasp depth
                if self.g_depth > tar_depth:
                    return None
                if self.g_depth == tar_depth:
                    release_only  = True
                else:
                    release_only = False
            else:
                if cur_depth >= tar_depth:                      # Expand to tree depth
                    return None
                else:
                    release_only = False
            
        if self.mediocrity > TreeNode.mediocrity_tolerance:     # Too Mediocre
            return None
        
        if not self.expanded:
            release_actions = []
            grasp_actions = []
            if self.actions is None:
                self.actions = self.gmrc.get_all_actions()
            for action in self.actions:
                if not isinstance(action, tuple):               # Release
                    if self.release_expanded:
                        continue
                    if not self.cgf_manager.can_release[action]:
                        continue
                    release_actions.append(action)
                else:                                           # Grasp
                    if release_only:
                        continue
                    if self.cgf_manager.is_cursed:
                        for curse_action in self.cgf_manager.curse:
                            gf = curse_action[0]
                            gt = curse_action[1]
                            idx = curse_action[2]
                            if not (gf == action[0] and gt == action[1]):
                                continue
                            grasp_actions.append((action, idx))
                    else:
                        grasp_actions.append(action)

            # Get Release Children
            release_children = [TreeNode.get_release_child(self, action)
                                for action in release_actions]
            for child in release_children:
                if child is not None:
                    self.children.append(child)

            # Get Grasp Children
            if self.cgf_manager.is_cursed:
                # Have to Lift the Curse
                grasp_children = [TreeNode.get_uncursed_child(self, action)
                                    for action in grasp_actions]
            else:
                # Try all feasible mediocre or constructive actions
                grasp_children = TreeNode.get_grasp_children(self, grasp_actions)
            for child in grasp_children:
                if child is not None:
                    self.children.append(child)
        
        self.release_expanded = True
        if not release_only:
            self.expanded = True

        for child in self.children:
            if child.is_goal():
                return child
            grand_child = child.expand_to(tar_depth, is_g_depth, cur_depth + 1)
            if grand_child is not None:                         # Early Stop
                return grand_child

        return None

    def expand(self, extra_depth: int = 1, is_g_depth = True):
        if is_g_depth:
            return self.expand_to(self.g_depth + extra_depth, is_g_depth)
        else:
            return self.expand_to(extra_depth, is_g_depth)

    # Whether this node contains all goal configuration angles
    def is_goal(self):
        if not len(self.cgf_manager.survival_idx) <= 0:
            return False
        if not self.tree.is_target_GMRC(self.get_id_ethnicity()[0]):
            return False
        return True
    
    # Release grasps that does not appear in goal configuration
    def extend_to_goal(self):
        if self.gmrc.w == self.tree.target_gmrc.w:
            if self.gmrc.v == self.tree.target_gmrc.v:
                return self
        for grip in range(self.gmrc.w + self.gmrc.v):
            if self.gmrc.is_grip_w[grip]:
                gripper = self.gmrc.gripper2module[3 * grip + 2]
            else:
                gripper = self.gmrc.gripper2module[3 * grip + 1]
            gripper_t = self.cgf_manager.correspondence[gripper]
            ht = gripper_t % 2
            mdl = gripper_t // 2
            if self.tree.target_gmrc.module2gripper[ht][mdl] < 0:
                child_gmrc = self.gmrc.copy()
                child_gmrc.execute_action(gripper)
                if not TreeNode.is_grouping:
                    child_group_feature = None
                else:
                    child_group_feature = (0, 0)
                child = TreeNode(child_gmrc, self.cgf_manager.copy(),
                                 self, self.g_depth, self.mediocrity,
                                 self.tree, child_group_feature)
                break
        return child.extend_to_goal()
    
    # Get the child of the node after a release action
    @staticmethod
    def get_release_child(node: TreeNode, release_action):
        new_gmrc = node.gmrc.copy()
        new_gmrc.execute_action(release_action)
        if not TreeNode.is_grouping:
            child_group_feature = None
        else:
            child_group_feature = (0, 0)
        new_node = TreeNode(new_gmrc, node.cgf_manager.copy(),
                            node, node.g_depth, node.mediocrity,
                            node.tree, child_group_feature)
        if not new_node.is_novel:
            return None
        return new_node

    # Get uncursed child of the node after a curse-lifting action
    @staticmethod
    def get_uncursed_child(node: TreeNode, curse_lifting_action):
        new_gmrc = node.gmrc.copy()
        grasp_action, idx = curse_lifting_action
        gf = grasp_action[0]
        gt = grasp_action[1]
        if not new_gmrc.execute_action(grasp_action):
            return None
        new_cgf_manager = node.cgf_manager.copy()
        new_cgf_manager.break_curse()
        new_cgf_manager.get_angle(gf, gt, idx)
        if not TreeNode.is_grouping:
            child_group_feature = None
        else:
            # The grip that is being constructed by this action
            built_grip = new_cgf_manager.Gamma_final[idx][5]
            child_group_feature = (1, built_grip)
        new_node = TreeNode(new_gmrc, new_cgf_manager,
                    node, node.g_depth + 1, 0,
                    node.tree, child_group_feature)
        if not new_node.is_novel:
            return None
        return new_node

    # Get grasp children of the node after all grasp actions
    @staticmethod
    def get_grasp_children(node: TreeNode, grasp_actions):
        children = []
        mmm_result = ParOptimizer.par_dock_min_max(repeat(node.gmrc),
                                                   grasp_actions,
                                                   TreeNode.max_workers)
        mdf_candidate = []
        for i, (mid_gmrc,
                (min_gmrc, min_ang),
                (max_gmrc, max_ang)) in enumerate(mmm_result):
            action = grasp_actions[i]
            if mid_gmrc is None:
                continue
            if not TreeNode.is_grouping:
                child_group_feature = None
            else:
                child_group_feature = (2, 0)
            mid_node = TreeNode(mid_gmrc, node.cgf_manager.copy(),
                            node, node.g_depth + 1, node.mediocrity + 1,
                            node.tree, child_group_feature)
            if mid_node.is_novel:
                children.append(mid_node)

            gf = action[0]
            gt = action[1]
            grip = mid_gmrc.module2gripper[gf % 2][gf // 2] // 3
            is_w_grip = mid_gmrc.is_grip_w[grip]
            mid_ang = mid_gmrc.get_grip_gamma(grip)

            if mid_node.is_novel:
                all_idxes = node.cgf_manager.get_angle_idxes(gf, gt, is_w_grip)
                for idx in all_idxes:
                    ang = node.cgf_manager.Gamma_final[idx][0]
                    if np.abs(ang - mid_ang) < 0.5 / 180 * np.pi:
                        bc_cgf_manager = node.cgf_manager.copy()
                        bc_cgf_manager.get_angle(gf, gt, idx)
                        bc_gmrc = mid_gmrc.copy()
                        if not TreeNode.is_grouping:
                            child_group_feature = None
                        else:
                            built_grip = bc_cgf_manager.Gamma_final[idx][5]
                            child_group_feature = (1, built_grip)
                        bc_node = TreeNode(bc_gmrc, bc_cgf_manager,
                                            node, node.g_depth + 1, 0,
                                            node.tree, child_group_feature)
                        children.append(bc_node)
            
            if mid_gmrc.is_2_cycle(-1):
                continue

            if min_gmrc is not None:
                if not TreeNode.is_grouping:
                    child_group_feature = None
                else:
                    child_group_feature = (2, 0)
                min_node = TreeNode(min_gmrc, node.cgf_manager.copy(),
                            node, node.g_depth + 1, node.mediocrity + 1,
                            node.tree, child_group_feature)
                if mid_node.is_novel:
                    children.append(min_node)

            if max_gmrc is not None:
                if not TreeNode.is_grouping:
                    child_group_feature = None
                else:
                    child_group_feature = (2, 0)
                max_node = TreeNode(max_gmrc, node.cgf_manager.copy(),
                            node, node.g_depth + 1, node.mediocrity + 1,
                            node.tree, child_group_feature)
                if max_node.is_novel:
                    children.append(max_node)

            mdf_candidate.append((mid_gmrc, action, min_ang, max_ang))

        # (gmrcs, grips, angles, choices, actions)
        mdf_nominee = ([], [], [], [], [])
        for mid_gmrc, action, min_ang, max_ang in mdf_candidate:
            gf = action[0]
            gt = action[1]
            grip = mid_gmrc.module2gripper[gf % 2][gf // 2] // 3
            is_w_grip = mid_gmrc.is_grip_w[grip]
            if is_w_grip:
                gi = mid_gmrc.gripper2module[3 * grip]
                gamma_0 = mid_gmrc.grsp_angs[3 * grip]
            else:
                gi = None
                gamma_0 = None
            all_choices = node.cgf_manager.get_ang_choices(
                gf, gt, is_w_grip, gi, gamma_0)
            for choice in all_choices:
                if isinstance(choice, tuple):
                    ang = choice[0][0]
                else:
                    temp_cgf_manager = node.cgf_manager.copy()
                    ang = temp_cgf_manager.get_angle(gf, gt, choice)
                    mid_gmrc.modify_ang_forcibly(ang, grip)
                    if node.tree.is_duplicated_gmrc(mid_gmrc,
                                                    temp_cgf_manager):
                        ang = np.inf
                    mid_gmrc.restore_grsp_ang()
                if ang >= min_ang and ang <= max_ang:
                    mdf_nominee[0].append(mid_gmrc)
                    mdf_nominee[1].append(grip)
                    mdf_nominee[2].append(ang)
                    mdf_nominee[3].append(choice)
                    mdf_nominee[4].append(action)
        
        modified_gmrcs = ParOptimizer.par_modify(mdf_nominee[0],
                                                 mdf_nominee[1],
                                                 mdf_nominee[2],
                                                 TreeNode.max_workers)
        for i, mdf_gmrc in enumerate(modified_gmrcs):
            choice = mdf_nominee[3][i]
            action = mdf_nominee[4][i]
            if mdf_gmrc is None:
                continue
            if isinstance(choice, tuple):
                # "ss" means stepping-stone here
                ss_cgf_manager = node.cgf_manager.copy()
                ss_cgf_manager.cursed_by(choice[1])
                if not TreeNode.is_grouping:
                    child_group_feature = None
                else:
                    # The grip that is being constructed by this action
                    built_grip = choice[0][1]
                    child_group_feature = (3, built_grip)
                ss_node = TreeNode(mdf_gmrc, ss_cgf_manager,
                                   node, node.g_depth + 1, 0,
                                   node.tree, child_group_feature)
                if ss_node.is_novel:
                    children.append(ss_node)
            else:
                # "bc" means building correspondence here
                bc_cgf_manager = node.cgf_manager.copy()
                gf = action[0]
                gt = action[1]
                idx = choice
                bc_cgf_manager.get_angle(gf, gt, idx)
                if not TreeNode.is_grouping:
                    child_group_feature = None
                else:
                    # The grip that is being constructed by this action
                    built_grip = bc_cgf_manager.Gamma_final[idx][5]
                    child_group_feature = (1, built_grip)
                bc_node = TreeNode(mdf_gmrc, bc_cgf_manager,
                                   node, node.g_depth + 1, 0,
                                   node.tree, child_group_feature)
                if bc_node.is_novel:
                    children.append(bc_node)
        
        return children

class Tree:
    suppress_print = False

    def __init__(self, gmrc: GMRC, cgf_manager: CGFManager, tar_gmrc: GMRC,
                 ed_estimator: EDEstimator, id_verdict: IDVerdict):
        self.nodes_at_depth: list[TreeNode] = [[]]
        self.ethinicity2id = dict()

        self.max_g_depth = 0
        self.target_gmrc = tar_gmrc
        # The number of constructive grasps to take for building target configuration
        self.num_constructive_grasp = tar_gmrc.w * 2 + tar_gmrc.v

        self.ed_estimator = ed_estimator
        self.id_verdict = id_verdict

        self.target_gmrc_id = self.id_verdict.get_identity(tar_gmrc)

        if not TreeNode.is_grouping:
            child_group_feature = None
        else:
            child_group_feature = (0, 0)
        self.root = TreeNode(gmrc,
                        cgf_manager = cgf_manager,
                        parent = None,
                        g_depth = 0,
                        mediocrity = 0,
                        tree = self,
                        group_feature = child_group_feature)
        
    # Decide whether a GMRC shape is target GMRC shape
    def is_target_GMRC(self, id: torch.tensor):
        return self.id_verdict.is_identical(id, self.target_gmrc_id)

    # Decide whether a GMRC shape has been reached by the tree or not
    def is_duplicated_gmrc(self, gmrc: GMRC, cgf_manager: CGFManager):
        gmrc_id, ethnicity = self.id_verdict.get_id_ethnicity(gmrc, cgf_manager)
        if not ethnicity in self.ethinicity2id:
            return False
        for id in self.ethinicity2id[ethnicity]:
            if self.id_verdict.is_identical(id, gmrc_id):
                return True
        return False

    def add_node_to_depth(self, node: TreeNode):
        skip = False
        temp = node
        while temp is not None:
            if temp.cgf_manager.is_cursed:
                skip = True
                break
            if temp.g_depth < node.g_depth:
                break
            temp = temp.parent
        if not skip:                    # Skip cursed or new-relieved-from-curse nodes
            node_id, ethnicity = node.get_id_ethnicity()
            if not ethnicity in self.ethinicity2id:
                self.ethinicity2id[ethnicity] = [node_id]
            else:
                for id in self.ethinicity2id[ethnicity]:
                    if self.id_verdict.is_identical(id, node_id):
                        return False                                # Prevent revisiting

        while node.g_depth > self.max_g_depth:
            self.nodes_at_depth.append([])
            self.max_g_depth = self.max_g_depth + 1
        self.nodes_at_depth[node.g_depth].append(node)
        return True

    # Traditional BFS that pushs the front by 1 g-depth with fixed mediocrity tolerance
    def push_front(self, time_budget):
        goal_node = None
        max_g_depth_before = self.max_g_depth
        start_time = time.time()
        for node in self.nodes_at_depth[-1]:
            goal_node = node.expand()
            if goal_node is not None:
                goal_node = goal_node.extend_to_goal()
                break
            if time.time() - start_time > time_budget:
                raise RuntimeError("Running Out of Time Budget!")
        max_g_depth_after = self.max_g_depth
        if max_g_depth_before == max_g_depth_after:
            raise RuntimeError("Can Not Further Expand Any Leaf Nodes!")
        num_new_nodes = len(self.nodes_at_depth[-1])
        if not Tree.suppress_print:
            print(f"Find {num_new_nodes} nodes at depth {len(self.nodes_at_depth) - 1}")
        return goal_node
    
    # Explore the configuration tree by gradually increasing mediocrity tolerance
    def explore(self, time_budget):
        TreeNode.mediocrity_tolerance = 0
        current_depth = 0
        if not Tree.suppress_print:
            print(f"\33[93mMediocrity Tolerance: {TreeNode.mediocrity_tolerance}\33[0m")
        start_time = time.time()
        while True:
            num_nodes = len(self.nodes_at_depth[current_depth])
            if not Tree.suppress_print:
                print(f"    Expanding {num_nodes} nodes at depth {current_depth}", 
                    end = "")
            for node in self.nodes_at_depth[current_depth]:
                goal_node = node.expand()
                if goal_node is not None:
                    goal_node = goal_node.extend_to_goal()
                    num_nodes = len(self.nodes_at_depth[current_depth])
                    if not Tree.suppress_print:
                        print(f"\r    Find {num_nodes} nodes at depth {current_depth}", 
                            end = "        \n")
                    current_depth = current_depth + 1
                    num_nodes = len(self.nodes_at_depth[current_depth])
                    if not Tree.suppress_print:
                        print(f"\r    Find {num_nodes} nodes at depth {current_depth}", 
                            end = "        \n")
                    return goal_node
                if time.time() - start_time > time_budget:
                    return None
            num_nodes = len(self.nodes_at_depth[current_depth])
            if not Tree.suppress_print:
                print(f"\r    Find {num_nodes} nodes at depth {current_depth}",
                    end = "        \n")
            current_depth = current_depth + 1
            if current_depth >= len(self.nodes_at_depth):
                TreeNode.mediocrity_tolerance = TreeNode.mediocrity_tolerance + 1
                current_depth = 0
                cur_mt = TreeNode.mediocrity_tolerance
                if not Tree.suppress_print:
                    print(f"\33[93mMediocrity Tolerance: {cur_mt}\33[0m")

# Edit Distance Estimator
class EDEstimator:
    def __init__(self,
                 eg: GENN.GENN = None,
                 ee: GENN.DegreeEmbedding = None,
                 ep: GENN.SequentialPooling = None,
                 device: torch.device = None):
        self.genn = eg
        self.genn_embedding = ee
        self.genn_pooling = ep
        self.device = device

    def get_distance(self, gmrc_1: GMRC, gmrc_2: GMRC):
        x_1, edge_index_1, cyclic_neighbors_1, neighbor_num_1 = \
            EMRC.get_representation(gmrc_1)
        x_2, edge_index_2, cyclic_neighbors_2, neighbor_num_2 = \
            EMRC.get_representation(gmrc_2)
        
        x_oh_1 = F.one_hot(x_1, GENN.GENN.max_num_degree).float()
        x_oh_2 = F.one_hot(x_2, GENN.GENN.max_num_degree).float()
        
        x_degree_feat_1 = self.genn_embedding(x_oh_1)
        x_degree_feat_2 = self.genn_embedding(x_oh_2)

        x_gnnout_feat_1 = self.genn(
            x_degree_feat_1,
            edge_index_1,
            cyclic_neighbors_1,
            neighbor_num_1)
        x_gnnout_feat_2 = self.genn(
            x_degree_feat_2, edge_index_2, cyclic_neighbors_2, neighbor_num_2)

        graph_feat_1 = self.genn_pooling(x_gnnout_feat_1, torch.tensor([x_1.size()[0]]))
        graph_feat_2 = self.genn_pooling(x_gnnout_feat_2, torch.tensor([x_2.size()[0]]))

        graph_feat_diff = graph_feat_1 - graph_feat_2
        distance = graph_feat_diff.norm(p=2, dim=-1)

        return distance

# Identity Verdict
class IDVerdict:
    strict_mode = False
    thd = 1e-7

    def __init__(self,
                 gg: GGNN.GGNN = None,
                 ge: GGNN.DegreeEmbedding = None,
                 gp: GGNN.SequentialPooling = None,
                 device: torch.device = None):
        self.ggnn = gg
        self.ggnn_embedding = ge
        self.ggnn_pooling = gp
        self.device = device

    # Get the identifier (hash value) using GGNN
    def get_identity(self, gmrc: GMRC):
        x, edge_index, cyclic_neighbors, neighbor_phis, neighbor_num = \
            gmrc.get_representation()
        with torch.inference_mode():
            x_oh = F.one_hot(x.to(self.device), GGNN.GGNN.max_num_degree).float()

            x_degree_feat = self.ggnn_embedding(x_oh)

            x_gnnout_feat = self.ggnn(
                x_degree_feat,
                edge_index.to(self.device),
                cyclic_neighbors.to(self.device),
                neighbor_phis.to(self.device),
                neighbor_num.to(self.device))

            graph_feat = self.ggnn_pooling(x_gnnout_feat, torch.tensor([x.size()[0]]))

        return graph_feat

    # Get the identifier and ethnicity of a (gmrc, cgf_manager) pair
    def get_id_ethnicity(self, gmrc: GMRC, cgf_manager: CGFManager):
        graph_feat = self.get_identity(gmrc)

        grsp_ang_parity = int(np.sum(np.abs(gmrc.grsp_angs)) / np.pi * 180)

        if IDVerdict.strict_mode:
            return (graph_feat, (grsp_ang_parity, int(cgf_manager.built_grip_parity)))
        else:
            return (graph_feat, grsp_ang_parity)

    def is_identical(self, id_1: torch.tensor, id_2: torch.tensor):
        graph_feat_diff = id_1 - id_2
        distance = graph_feat_diff.norm(p=2, dim=-1)
        if distance >= IDVerdict.thd:
            return False        
        return True
    
# Monte Carlo Tree Node
class MCTreeNode:
    def __init__(self, node: TreeNode, parent: MCTreeNode = None,
                 tree: MCTree = None):
        self.node = node                                            # Actual tree node
        self.parent = parent                                        # Parent MCTreeNode
        if tree is None:
            self.tree = self.parent.tree                            # Belonging MCTree
        else:
            self.tree = tree
        self.children: list[MCTreeNode] = []                        # Children MCTreeNode
        self.survival_children: list[int] = []                      # Alive Children
        self.n = 0.0                                                # Times of selected
        self.Q = 0.0                                                # Estimated node value
        # NOTE: If not expanded, the node is actually not added in the MCTree
        self.is_expanded = False                                    # Whether expanded
        if self.node is self.tree.goal_node:
            self.is_goal = True                                     # Whether is goal
        else:
            self.is_goal = False

    def __str__(self):
        return f"Type: N, GF: {self.node.group_feature}; Q: {self.Q}; N: {self.n}"

    def __repr__(self):
        return self.__str__()

    # Get first-play urgency of this node
    # NOTE: This is pretty domain-specific
    def get_FPU(self):
        constructed = self.node.cgf_manager.num_constructed
        to_construct = self.node.tree.num_constructive_grasp
        if self.node.g_depth > 0:
            efficiency = constructed / self.node.g_depth
        else:
            efficiency = 1.0
        progress = constructed / to_construct * efficiency
        if self.node.group_feature[0] == 0:                         # From releasing
            """ Discourage Exploration when all in a Releasing Group """
            pseudo_Q = MCTree.w_progress * progress + \
                MCTree.w_promising * MCTree.promising_score_release
            return pseudo_Q
        elif self.node.group_feature[0] == 1:                       # From constructing
            """ Encourage Exploration to Find Best Constructive Action """
            pseudo_Q = MCTree.w_progress * progress + \
                MCTree.w_promising * MCTree.promising_score_constructive
            pseudo_n = to_construct - constructed + 1
            return pseudo_Q + MCTree.c * np.sqrt(np.log(pseudo_n)/pseudo_n)
        elif self.node.group_feature[0] == 3:                       # From Curse
            """Encourage Exploration to Find Best Stepping-Stone (Cursed) Action"""
            pseudo_Q = MCTree.w_progress * progress + \
                MCTree.w_promising * MCTree.promising_score_constructive
            pseudo_n = to_construct - constructed + 1
            return pseudo_Q + MCTree.c * np.sqrt(np.log(pseudo_n)/pseudo_n)
        else:                                                       # From mediocre
            """ Discourage Exploration when all in a Mediocre Group """
            pseudo_Q = MCTree.w_progress * progress + \
                MCTree.w_promising * MCTree.promising_score_mediocre
            return pseudo_Q

    # N: Number of times that the parent of self has been selected
    def get_UCB(self, N):
        if self.n == 0:
            return self.get_FPU()
        return self.Q + MCTree.c * np.sqrt(np.log(N) / self.n)

    # The expand() function is also serving as part of the heuristic function
    def expand(self):
        goal_node = self.node.expand(1, False)
        if goal_node is not None:
            self.tree.is_goal_found = True
            self.tree.goal_node = goal_node
        group2node = {}
        num_groups = 0
        for child_node in self.node.children:
            g = child_node.group_feature[0]
            if not g in group2node:
                group2node[g] = [child_node]
                num_groups = num_groups + 1
            else:
                group2node[g].append(child_node)
        if num_groups == 1:
            for child_node in self.node.children:
                self.children.append(MCTreeNode(child_node, self))
        else:
            for group in group2node:
                self.children.append(MCTreeGroupNode(
                    group2node[group],
                    self,
                    0
                ))
            for child in self.children:
                child.expand()  # Recursively expand all group nodes

        self.survival_children = list(range(len(self.children)))
        self.is_expanded = True

    # Select a node that 1. leads to a leaf node or 2. to be expanded and added
    def select(self) -> MCTreeNode:
        ucb_values = []
        for i in self.survival_children:
            ucb_values.append(self.children[i].get_UCB(self.n))
        ucb_array = np.array(ucb_values)
        child = self.children[self.survival_children[np.argmax(ucb_array)]]
        return child

    # Instead of roll-outs, a heuristic is used instead for evaluating the node
    #   NOTE: This method is domain-specific (related to actual tree nodes)
    #   NOTE:   Here, the heuristic value is strictly withhin [0, 1]
    def simulate(self):
        num_releasing = 0
        num_constructive = 0
        num_curse = 0
        num_mediocre = 0
        if len(self.children) == 0:
            return 0
        if not isinstance(self.children[0], MCTreeGroupNode):
            if self.children[0].node.group_feature[0] == 0:
                num_releasing = len(self.children)
            elif self.children[0].node.group_feature[0] == 1:
                num_constructive = len(self.children)
            elif self.children[0].node.group_feature[0] == 2:
                num_mediocre = len(self.children)
            elif self.children[0].node.group_feature[0] == 3:
                num_curse = len(self.children)
        else:
            for child in self.children:
                assert isinstance(child, MCTreeGroupNode)
                if child.group_feature[0] == 0:
                    num_releasing = len(child.nodes)
                elif child.group_feature[0] == 1:
                    num_constructive = len(child.nodes)
                elif child.group_feature[0] == 2:
                    num_mediocre = len(child.nodes)
                elif child.group_feature[0] == 3:
                    num_curse = len(child.nodes)

        constructed = self.node.cgf_manager.num_constructed
        to_construct = self.node.tree.num_constructive_grasp
        if self.node.g_depth > 0:
            efficiency = constructed / self.node.g_depth
        else:
            efficiency = 1.0
        progress_score =  constructed / to_construct * efficiency

        if num_constructive == 0:
            if num_releasing > 0:
                promising_score = MCTree.promising_score_release
            elif num_curse > 0:
                promising_score = MCTree.promising_score_curse
            elif num_mediocre > 0:
                promising_score = MCTree.promising_score_mediocre
            else:
                return 0
        else:
            promising_score = MCTree.promising_score_constructive
        
        w_progress = MCTree.w_progress
        w_future = MCTree.w_promising

        return w_progress * progress_score + w_future * promising_score

    def backpropagate(self, Q):
        self.Q = (self.Q * self.n + Q) / (self.n + 1)
        self.n = self.n + 1
        if self.parent is not None:
            if len(self.survival_children) <= 0:
                parent_new_survival_children = []
                for i in self.parent.survival_children:
                    if not self.parent.children[i] is self:
                        parent_new_survival_children.append(i)
                self.parent.survival_children = parent_new_survival_children
            self.parent.backpropagate(Q)

# Monte Carlo Tree Group Node
class MCTreeGroupNode(MCTreeNode):
    def __init__(self, nodes: list[TreeNode], parent: MCTreeNode, 
                 group_level: int):
        super().__init__(None, parent)
        self.nodes = nodes
        self.num_nodes = len(self.nodes)
        self.group_feature = self.nodes[0].group_feature
        self.group_level = group_level

    def __str__(self):
        return f"Type: G, GF: {self.group_feature}; Q: {self.Q}; N: {self.n}"
    
    def get_FPU(self):
        constructed = self.nodes[0].cgf_manager.num_constructed
        to_construct = self.nodes[0].tree.num_constructive_grasp
        if self.nodes[0].g_depth > 0:
            efficiency = constructed / self.nodes[0].g_depth
        else:
            efficiency = 1.0
        progress = constructed / to_construct * efficiency
        if self.group_level == 0:
            if self.group_feature[0] == 0:                          # Releasing
                """ Encourage Releasing First to Prepare for Constructing """
                pseudo_Q = MCTree.w_progress * progress + \
                    MCTree.w_promising * MCTree.promising_score_constructive
                return pseudo_Q + MCTree.c * np.sqrt(np.log(3))
            elif self.group_feature[0] == 1:                        # Constructive
                """ Encourage Constructive Actions, but Weaker than Releasing """
                pseudo_Q = MCTree.w_progress * progress + \
                    MCTree.w_promising * MCTree.promising_score_constructive
                return pseudo_Q
            elif self.group_feature[0] == 3:
                """Discourage Trying Cursed Actions"""              # Cursed
                pseudo_Q = MCTree.w_progress * progress + \
                    MCTree.w_promising * MCTree.promising_score_curse
                return pseudo_Q
            else:                                                   # Mediocre
                """ Discourage Trying Mediocre Actions """
                return MCTree.w_promising * MCTree.promising_score_mediocre
        else:                                                       # Grip Groups 
            """ Encourage Trying Different Correspondence Sequence """
            pseudo_Q = MCTree.w_progress * progress + \
                MCTree.w_promising * MCTree.promising_score_constructive
            return pseudo_Q + MCTree.c * np.sqrt(np.log(self.parent.n + 1))

    # Expand to 1. more group nodes or 2. concrete nodes
    def expand(self):
        subgroup2node = {}
        num_subgroup = 0
        for node in self.nodes:
            if self.group_level + 1 < len(node.group_feature):
                g = node.group_feature[self.group_level + 1]
            else:
                g = None
            if not g in subgroup2node:
                subgroup2node[g] = [node]
                num_subgroup = num_subgroup + 1
            else:
                subgroup2node[g].append(node)
        
        if num_subgroup == 1:
            for node in self.nodes:
                self.children.append(MCTreeNode(node, self))
        else:
            for subgroup in subgroup2node:
                self.children.append(MCTreeGroupNode(
                    subgroup2node[subgroup],
                    self,
                    self.group_level + 1
                ))
            for child in self.children:
                child.expand()                      # Recursively expand all group nodes

        self.survival_children = list(range(len(self.children)))
        self.is_expanded = True

# Monte Carlo Tree
class MCTree:
    c = 0.2                                         # UCB Constant

    w_progress = 0.1
    w_promising = 0.9

    promising_score_constructive = 1.0
    promising_score_release = 0.7
    promising_score_curse = 0.3
    promising_score_mediocre = 0.1

    def __init__(self, node: TreeNode):
        self.is_goal_found = False
        self.goal_node: TreeNode = None
        self.root = MCTreeNode(node, None, self)

        self.root.expand()                          # Expand and add to tree

    def select(self):                               # Select a node
        node = self.root.select()
        while node.is_expanded:
            new_node = node.select()
            while new_node is None:
                new_node = node.select()
            node = new_node
        return node

    def search_for_goal(self, time_budget):
        start_time = time.time()
        while True:
            node = self.select()
            node.expand()
            Q = node.simulate()
            node.backpropagate(Q)

            if self.is_goal_found:
                return self.goal_node.extend_to_goal()
            if time.time() - start_time > time_budget:
                return None

class CMRRP:
    def __init__(self,
                 gg: GGNN.GGNN = None,
                 ge: GGNN.DegreeEmbedding = None,
                 gp: GGNN.SequentialPooling = None,
                 eg: GENN.GENN = None,
                 ee: GENN.DegreeEmbedding = None,
                 ep: GENN.SequentialPooling = None,
                 device = None
                 ):
        self.ed_estimator = EDEstimator(eg, ee, ep, device)
        self.id_verdict = IDVerdict(gg, ge, gp, device)
        self.tree = None

    """
    Methods:
        BFS: Simple breadth first search with fixed mediocrity tolerance
        IMT_BFS: BFS with Iterative Mediocrity Tolerance
        MCTS: Single Monte Carlo Tree Search with Hierarchical Grouping Nodes
    """
    def plan(self, gmrc_1: GMRC, gmrc_2: GMRC, method = "BFS",
             time_budget = 300, is_print = True):
        GMRC.suppress_action_err = True
        assert gmrc_1.m == gmrc_2.m

        IDVerdict.strict_mode = False
        target_angles = dict()
        for grip in range(len(gmrc_2.grippers) // 3):
            if gmrc_2.is_grip_w[grip]:
                gpr_list = [grip * 3, grip * 3 + 1, grip * 3 + 2]
            else:
                gpr_list = [grip * 3]
            for gpr in gpr_list:
                ang = int(1e2 * np.abs(gmrc_2.grsp_angs[gpr] / np.pi * 180))
                if ang in target_angles and not target_angles[ang] == grip:
                    IDVerdict.strict_mode = True
                target_angles[ang] = grip
        if IDVerdict.strict_mode and is_print:
            print("\033[95mDetected Duplicated Angles, Turning on IDVerdict Strict Mode\033[0m")
        Tree.suppress_print = not is_print

        CGFManager.m = gmrc_1.m
        cgf_manager = CGFManager(gmrc_2.get_Gamma_final())
        self.tree = Tree(gmrc_1, cgf_manager, gmrc_2, self.ed_estimator, self.id_verdict)

        if method == "BFS":
            TreeNode.is_grouping = False
            while True:
                node = self.tree.push_front(time_budget)
                if node is not None:
                    break
        elif method == "IMT_BFS":
            TreeNode.is_grouping = False
            node = self.tree.explore(time_budget)
            if node is None:
                return None
        elif method == "MCTS":
            TreeNode.mediocrity_tolerance = 9999
            TreeNode.is_grouping = True
            self.mctree = MCTree(self.tree.root)
            node = self.mctree.search_for_goal(time_budget)
            if node is None:
                return None

        path = [node]
        while node.parent is not None:
            node = node.parent
            path.append(node)
        path.reverse()
        return path