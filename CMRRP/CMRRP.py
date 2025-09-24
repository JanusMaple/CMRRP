"""
Continuum Modular Robot Reconfigure Planner: Task and Motion Planning
"""

from __future__ import annotations
import sys
sys.path.append('..')
sys.path.append('../GENN')
sys.path.append('../GGNN')
import copy
import warnings
import torch
import torch.nn.functional as F
import numpy as np
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
                 can_release = None, is_cursed = False, curse = None):
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
            built_grip = self.Gamma_final[idx][1] // 3
            key = (gamma, built_grip)
            if not key in cursed_choices:
                cursed_choices[key] = []
            cursed_choices[key].append((emt_new_gf, emt_new_gt, idx, new_gf))
        idxes = self.get_angle_idxes(emt_new_gt, emt_new_gf, False)
        for idx in idxes:
            gamma = self.Gamma_final[idx][0]
            built_grip = self.Gamma_final[idx][1] // 3
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
                built_grip = self.Gamma_final[idx][1] // 3
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
                built_grip = self.Gamma_final[idx][1] // 3
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
                          curse=copy.deepcopy(self.curse))

# A search tree node contains: 1. a unique gmrc shape; 2. a partial correspondence
class TreeNode:
    mediocrity_tolerance = 0
    is_grouping = False

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
            self.id_ethnicity = self.tree.id_verdict.get_id_ethnicity(self.gmrc)
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
            
        if self.mediocrity > TreeNode.mediocrity_tolerance:
            return None
        
        if not self.expanded:
            if self.actions is None:
                self.actions = self.gmrc.get_all_actions()
            for action in self.actions:
                new_gmrc = self.gmrc.copy()
                if not isinstance(action, tuple):               # Release
                    if self.release_expanded:
                        continue
                    if not self.cgf_manager.can_release[action]:
                        continue
                    new_gmrc.execute_action(action)
                    if not TreeNode.is_grouping:
                        child_group_feature = None
                    else:
                        child_group_feature = (0, 0)
                    new_node = TreeNode(new_gmrc, self.cgf_manager.copy(),
                                        self, self.g_depth, self.mediocrity,
                                        self.tree, child_group_feature)
                    if new_node.is_novel:
                        self.children.append(new_node)
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
                            if not new_gmrc.execute_action(action):
                                continue
                            new_cgf_manager = self.cgf_manager.copy()
                            new_cgf_manager.break_curse()
                            new_cgf_manager.get_angle(gf, gt, idx)
                            if not TreeNode.is_grouping:
                                child_group_feature = None
                            else:
                                # The grip that is being constructed by this action
                                built_grip = new_cgf_manager.correspondence[gf] // 3
                                child_group_feature = (1, built_grip)
                            new_node = TreeNode(new_gmrc, new_cgf_manager,
                                        self, self.g_depth + 1, 0,
                                        self.tree, child_group_feature)
                            if new_node.is_novel:
                                self.children.append(new_node)
                        continue
                    if not new_gmrc.execute_action(action):
                        continue
                    self.children.extend(
                        self._get_all_memebers_in_grasping_group(new_gmrc, action))
        
        self.release_expanded = True
        if not release_only:
            self.expanded = True

        self.children.reverse()
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
        if len(self.cgf_manager.survival_idx) <= 0:
            return True
        return False
    
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

    # Get all reasonable children from the same grasping action based on eldest sibling
    # NOTE: Will not have a->b with alpha and b->a with alpha
    #       since will only have a->b from EMRC.get_all_actions()
    def _get_all_memebers_in_grasping_group(self, new_gmrc: GMRC, action: tuple):
        if not TreeNode.is_grouping:
            child_group_feature = None
        else:
            child_group_feature = (2, 0)
        optim_node = TreeNode(new_gmrc, self.cgf_manager.copy(),
                            self, self.g_depth + 1, self.mediocrity + 1,
                            self.tree, child_group_feature)
        if optim_node.is_novel:
            members = [optim_node]
        else:
            members = []
        
        gf = action[0]
        gt = action[1]
        grip = new_gmrc.module2gripper[gf % 2][gf // 2] // 3
        is_w_grip = new_gmrc.is_grip_w[grip]
        mid_ang = new_gmrc.get_grip_gamma(grip)

        if new_gmrc.is_2_cycle(-1):
            if optim_node.is_novel:
                all_idxes = self.cgf_manager.get_angle_idxes(gf, gt, is_w_grip)
                for idx in all_idxes:
                    new_cgf_manager = self.cgf_manager.copy()
                    ang = new_cgf_manager.get_angle(gf, gt, idx)
                    if np.abs(ang - mid_ang) < 0.001 / 180 * np.pi:
                        optim_node.mediocrity = 0
                        optim_node.cgf_manager.get_angle(gf, gt, idx)
                        members.append(optim_node)
            return members

        min_gmrc = new_gmrc.copy()
        min_gmrc.modify_grsp_ang(grip, -GMRC.grsp_ang_cap)
        min_ang = min_gmrc.get_grip_gamma(grip)
        if min_ang < mid_ang:
            if not TreeNode.is_grouping:
                child_group_feature = None
            else:
                child_group_feature = (2, 0)
            min_node = TreeNode(min_gmrc, self.cgf_manager.copy(),
                                self, self.g_depth + 1, self.mediocrity + 1,
                                self.tree, child_group_feature)
            if min_node.is_novel:
                members.append(min_node)

        max_gmrc = new_gmrc.copy()
        max_gmrc.modify_grsp_ang(grip, GMRC.grsp_ang_cap)
        max_ang = max_gmrc.get_grip_gamma(grip)
        if max_ang > mid_ang:
            if not TreeNode.is_grouping:
                child_group_feature = None
            else:
                child_group_feature = (2, 0)
            max_node = TreeNode(max_gmrc, self.cgf_manager.copy(),
                                self, self.g_depth + 1, self.mediocrity + 1,
                                self.tree, child_group_feature)
            if max_node.is_novel:
                members.append(max_node)

        if is_w_grip:
            gi = new_gmrc.gripper2module[3 * grip]
            gamma_0 = new_gmrc.grsp_angs[3 * grip]
        else:
            gi = None
            gamma_0 = None
        all_choices = self.cgf_manager.get_ang_choices(gf, gt, is_w_grip, gi, gamma_0)
        for choice in all_choices:
            if isinstance(choice, tuple):
                # "ss" means stepping-stone here
                ss_cgf_manager = self.cgf_manager.copy()
                ss_cgf_manager.cursed_by(choice[1])
                ang = choice[0][0]
                if ang <= min_ang and ang >= max_ang:
                    continue
                ss_gmrc = new_gmrc.copy()
                if not ss_gmrc.modify_grsp_ang(grip, ang):
                    continue
                if np.abs(ss_gmrc.get_grip_gamma(grip) - ang) > 1e-3:
                    continue
                
                if not TreeNode.is_grouping:
                    child_group_feature = None
                else:
                    # The grip that is being constructed by this action
                    built_grip = choice[0][1]
                    child_group_feature = (1, built_grip)
                ss_node = TreeNode(ss_gmrc, ss_cgf_manager,
                                   self, self.g_depth + 1, 0,
                                   self.tree, child_group_feature)
                if ss_node.is_novel:
                    members.append(ss_node)
            else:                                       # Building some correspondence
                idx = choice
                # "bc" means building correspondence here
                bc_cgf_manager = self.cgf_manager.copy()
                ang = bc_cgf_manager.get_angle(gf, gt, idx)
                if ang <= min_ang and ang >= max_ang:
                    continue
                bc_gmrc = new_gmrc.copy()
                bc_gmrc.modify_ang_forcibly(ang, grip)
                if self.tree.is_duplicated_gmrc(bc_gmrc):
                    continue
                bc_gmrc.restore_grsp_ang()
                if not bc_gmrc.modify_grsp_ang(grip, ang):
                    continue
                if np.abs(bc_gmrc.get_grip_gamma(grip) - ang) > 1e-3:
                    continue
                if not TreeNode.is_grouping:
                    child_group_feature = None
                else:
                    # The grip that is being constructed by this action
                    built_grip = bc_cgf_manager.correspondence[gf] // 3
                    child_group_feature = (1, built_grip)
                bc_node = TreeNode(bc_gmrc, bc_cgf_manager,
                                self, self.g_depth + 1, 0,
                                self.tree, child_group_feature)
                if bc_node.is_novel:
                    members.append(bc_node)
                if bc_node.is_goal():
                    break

        return members

class Tree:
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

        self.target_id, _ = self.id_verdict.get_id_ethnicity(tar_gmrc)

    # Decide whether a GMRC shape has been reached by the tree or not
    def is_duplicated_gmrc(self, gmrc: GMRC):
        gmrc_id, ethnicity = self.id_verdict.get_id_ethnicity(gmrc)
        if not ethnicity in self.ethinicity2id:
            return False
        for id in self.ethinicity2id[ethnicity]:
            if self.id_verdict.is_identical(id, gmrc_id):
                return True
        return False

    def add_node_to_depth(self, node: TreeNode):
        if not node.cgf_manager.is_cursed:
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
    def push_front(self):
        goal_node = None
        max_g_depth_before = self.max_g_depth
        for node in self.nodes_at_depth[-1]:
            goal_node = node.expand()
            if goal_node is not None:
                goal_node = goal_node.extend_to_goal()
                break
        max_g_depth_after = self.max_g_depth
        if max_g_depth_before == max_g_depth_after:
            raise RuntimeError("Can not further expand any leaf nodes!")
        num_new_nodes = len(self.nodes_at_depth[-1])
        print(f"Find {num_new_nodes} nodes at depth {len(self.nodes_at_depth) - 1}")
        return goal_node
    
    # Explore the configuration tree by gradually increasing mediocrity tolerance
    def explore(self):
        TreeNode.mediocrity_tolerance = 0
        current_depth = 0
        print(f"\33[93mMediocrity Tolerance: {TreeNode.mediocrity_tolerance}\33[0m")
        while True:
            num_nodes = len(self.nodes_at_depth[current_depth])
            print(f"    Expanding {num_nodes} nodes at depth {current_depth}", 
                  end = "")
            for node in self.nodes_at_depth[current_depth]:
                goal_node = node.expand()
                if goal_node is not None:
                    goal_node = goal_node.extend_to_goal()
                    num_nodes = len(self.nodes_at_depth[current_depth])
                    print(f"\r    Find {num_nodes} nodes at depth {current_depth}", 
                          end = "        \n")
                    current_depth = current_depth + 1
                    num_nodes = len(self.nodes_at_depth[current_depth])
                    print(f"\r    Find {num_nodes} nodes at depth {current_depth}", 
                          end = "        \n")
                    return goal_node
            num_nodes = len(self.nodes_at_depth[current_depth])
            print(f"\r    Find {num_nodes} nodes at depth {current_depth}",
                  end = "        \n")
            current_depth = current_depth + 1
            if current_depth >= len(self.nodes_at_depth):
                TreeNode.mediocrity_tolerance = TreeNode.mediocrity_tolerance + 1
                current_depth = 0
                cur_mt = TreeNode.mediocrity_tolerance
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
    def get_id_ethnicity(self, gmrc):
        x, edge_index, cyclic_neighbors, neighbor_phis, neighbor_num = \
            gmrc.get_representation()
        
        x_oh = F.one_hot(x.to(self.device), GGNN.GGNN.max_num_degree).float()

        x_degree_feat = self.ggnn_embedding(x_oh)

        x_gnnout_feat = self.ggnn(
            x_degree_feat,
            edge_index.to(self.device),
            cyclic_neighbors.to(self.device),
            neighbor_phis.to(self.device),
            neighbor_num.to(self.device))

        graph_feat = self.ggnn_pooling(x_gnnout_feat, torch.tensor([x.size()[0]]))

        grsp_ang_parity = int(np.sum(np.abs(gmrc.grsp_angs)) / np.pi * 180)

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
            """ Disencourage Exploration when all in a Releasing Group """
            pseudo_Q = MCTree.w_progress * progress + \
                MCTree.w_promising * MCTree.promising_score_release
            return pseudo_Q
        elif self.node.group_feature[0] == 1:                       # From constructing
            """ Encourage Exploration to Find Best Constructive Action """
            pseudo_Q = MCTree.w_progress * progress + \
                MCTree.w_promising * MCTree.promising_score_constructive
            num_siblings = len(self.parent.children)
            return pseudo_Q + MCTree.c * np.sqrt(np.log(num_siblings))
        else:                                                       # From mediocre
            """ Disencourage Exploration when all in a Mediocre Group """
            pseudo_Q = MCTree.w_progress * progress + \
                MCTree.w_promising * MCTree.promising_score_mediocre
            return pseudo_Q

    # N: Number of times that the parent of self has been selected
    def get_UCB(self, N):
        if self.n == 0:
            return self.get_FPU()
        if self.Q == 0:
            return 0.0
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
        # p = torch.tensor(np.exp(ucb_array) / np.sum(np.exp(ucb_array)))
        # child = self.children[self.survival_children[torch.multinomial(p, 1).item()]]
        child = self.children[self.survival_children[np.argmax(ucb_array)]]
        return child

    # Instead of roll-outs, a heuristic is used instead for evaluating the node
    #   NOTE: This method is domain-specific (related to actual tree nodes)
    #   NOTE:   Here, the heuristic value is strictly withhin [0, 1]
    def simulate(self):
        num_releasing = 0
        num_constructive = 0
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
        else:
            for child in self.children:
                assert isinstance(child, MCTreeGroupNode)
                if child.group_feature[0] == 0:
                    num_releasing = len(self.children)
                elif child.group_feature[0] == 1:
                    num_constructive = len(self.children)
                elif child.group_feature[0] == 2:
                    num_mediocre = len(self.children)

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
            else:                                                   # Mediocre
                """ Disencourage Trying Mediocre Actions """
                return MCTree.w_promising * MCTree.promising_score_mediocre
        else:                                                       # Grip Groups 
            """ Encourage Trying Different Correspondence Sequence """
            pseudo_Q = MCTree.w_progress * progress + \
                MCTree.w_promising * MCTree.promising_score_constructive
            return pseudo_Q + MCTree.c * np.sqrt(np.log(self.parent.n + 1))
        
    # Get the number of concrete nodes within this group node
    def get_num_nodes(self):
        num = 0
        for child in self.children:
            if isinstance(child, MCTreeGroupNode):
                num = num + child.get_num_nodes()
            else:
                num = num + 1
        return num

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

    w_progress = 0.4
    w_promising = 0.7

    promising_score_constructive = 1.0
    promising_score_release = 0.6
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

    def search_for_goal(self):
        while True:
            node = self.select()
            node.expand()
            Q = node.simulate()
            node.backpropagate(Q)

            if self.is_goal_found:
                return self.goal_node.extend_to_goal()

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
        DMT_BFS: BFS with dynamic mediocrity tolerance
        MCTS: Single Monte Carlo Tree Search with Hierarchical Grouping Nodes
    """
    def plan(self, gmrc_1: GMRC, gmrc_2: GMRC, method = "BFS"):
        GMRC.suppress_action_err = True
        assert gmrc_1.m == gmrc_2.m
        CGFManager.m = gmrc_1.m
        cgf_manager = CGFManager(gmrc_2.get_Gamma_final())
        self.tree = Tree(gmrc_1, cgf_manager, gmrc_2, self.ed_estimator, self.id_verdict)

        if method == "BFS":
            TreeNode.is_grouping = False
            while True:
                node = self.tree.push_front()
                if node is not None:
                    break
        elif method == "DMT_BFS":
            TreeNode.is_grouping = False
            node = self.tree.explore()
        elif method == "MCTS":
            TreeNode.mediocrity_tolerance = 9999
            TreeNode.is_grouping = True
            self.mctree = MCTree(self.tree.root)
            node = self.mctree.search_for_goal()

        path = [node]
        while node.parent is not None:
            node = node.parent
            path.append(node)
        path.reverse()
        return path