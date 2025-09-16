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
                 correspondence: list = None, num_corresponded: int = 0,
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

        # Number of modules that have been corresponded
        self.num_corresponded = num_corresponded

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
            if not gamma in cursed_choices:
                cursed_choices[gamma] = []
            cursed_choices[gamma].append((emt_new_gf, emt_new_gt, idx, new_gf))
        idxes = self.get_angle_idxes(emt_new_gt, emt_new_gf, False)
        for idx in idxes:
            gamma = self.Gamma_final[idx][0]
            if not gamma in cursed_choices:
                cursed_choices[gamma] = []
            cursed_choices[gamma].append((emt_new_gt, emt_new_gf, idx, new_gf))
        if is_w_grip:
            emt_new_gi = 2 * (new_gi // 2) + 1 - new_gi % 2
            idxes = self.get_angle_idxes(emt_new_gf, emt_new_gi, False)
            for idx in idxes:
                if self.Gamma_final[idx][0] < 0:
                    gamma = -np.pi - gamma_0 - self.Gamma_final[idx][0]
                else:
                    gamma = np.pi - gamma_0 - self.Gamma_final[idx][0]
                if not gamma in cursed_choices:
                    cursed_choices[gamma] = []
                cursed_choices[gamma].append((emt_new_gf, emt_new_gi, idx, new_gf))
            idxes = self.get_angle_idxes(emt_new_gi, emt_new_gf, False)
            for idx in idxes:
                if self.Gamma_final[idx][0] < 0:
                    gamma = np.pi - gamma_0 - (-self.Gamma_final[idx][0])
                else:
                    gamma = -np.pi - gamma_0 - (-self.Gamma_final[idx][0])
                if not gamma in cursed_choices:
                    cursed_choices[gamma] = []
                cursed_choices[gamma].append((emt_new_gi, emt_new_gf, idx, new_gf))

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
                          num_corresponded=self.num_corresponded,
                          gf_idx_list=copy.deepcopy(self.gf_idx_list),
                          gt_idx_list=copy.deepcopy(self.gt_idx_list),
                          can_release=copy.deepcopy(self.can_release),
                          is_cursed=self.is_cursed,
                          curse=copy.deepcopy(self.curse))

# A search tree node contains: 1. a unique gmrc shape; 2. a partial correspondence
class TreeNode:
    mediocrity_tolerance = 0

    def __init__(self, gmrc: GMRC, cgf_manager: CGFManager, 
                 parent: TreeNode = None, g_depth: int = 0,
                 mediocrity: int = 0, tree: Tree = None):
        self.cgf_manager = cgf_manager      # Partial Correspondence
        self.gmrc: GMRC = gmrc              # GMRC

        self.parent: TreeNode = parent      # Parent Tree Node

        self.g_depth: int = g_depth         # Grasping Depth

        self.interesting_children: list[TreeNode] = []
        self.mediocre_children: list[TreeNode] = []

        self.mediocrity: int = mediocrity   # Current Mediocrity

        self.tree: Tree = tree              # Belonging Tree
        
        self.actions = None                 # All Potential Actions

        self.release_expanded = False       # Whether have expanded releasing children
        self.expanded = False               # Whether have expanded all children

        self.id_ethnicity = None            # Tuple of node id and ethnicity

        self.is_novel = self.tree.add_node_to_depth(self)

    # Get both node identifier and its ethnicity (based on grsp_ang parity)
    def get_id_ethnicity(self):
        if self.id_ethnicity is None:
            self.id_ethnicity = self.tree.id_verdict.get_id_ethnicity(self.gmrc)
        return self.id_ethnicity
        
    def expand_to(self, tar_g_depth = None):
        if tar_g_depth is not None:
            if self.g_depth > tar_g_depth:
                return None
            if self.g_depth == tar_g_depth:
                release_only  = True
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
                    new_node = TreeNode(new_gmrc, self.cgf_manager.copy(),
                                        self, self.g_depth, self.mediocrity, self.tree)
                    if new_node.is_novel:
                        self.interesting_children.append(new_node)
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
                            new_node = TreeNode(new_gmrc, new_cgf_manager,
                                        self, self.g_depth + 1, 0, self.tree)
                            if new_node.is_novel:
                                self.interesting_children.append(new_node)
                        continue
                    if not new_gmrc.execute_action(action):
                        continue
                    ims, mms = self._get_all_memebers_in_grasping_group(
                        new_gmrc, action)
                    self.interesting_children.extend(ims)
                    self.mediocre_children.extend(mms)
        
        self.release_expanded = True
        if not release_only:
            self.expanded = True

        self.interesting_children.reverse()
        for child in self.interesting_children:
            if child.is_goal():
                return child
            grand_child = child.expand_to(tar_g_depth)
            if grand_child is not None:
                return grand_child

        return None

    def expand(self, extra_depth: int = 1):
        return self.expand_to(self.g_depth + extra_depth)

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
                child = TreeNode(child_gmrc, self.cgf_manager.copy(),
                                 self, self.g_depth, 
                                 self.mediocrity, self.tree)
                break
        return child.extend_to_goal()

    # Get all reasonable children from the same grasping action based on eldest sibling
    # NOTE: Will not have a->b with alpha and b->a with alpha
    #       since will only have a->b from EMRC.get_all_actions()
    """
    Return (interesting_members: list[TreeNode], mediocre_members: list[TreeNode])
        interesting_members:    mediority <= mediority_tolerance
        mediocre_members:       mediority > mediority_tolerance
    """
    def _get_all_memebers_in_grasping_group(self, new_gmrc: GMRC, action: tuple):
        optim_node = TreeNode(new_gmrc, self.cgf_manager.copy(),
                            self, self.g_depth + 1, self.mediocrity + 1, self.tree)
        if optim_node.is_novel:
            if self.mediocrity < TreeNode.mediocrity_tolerance:
                interesting_members = [optim_node]
                mediocre_members = []
            else:
                interesting_members = []
                mediocre_members = [optim_node]
        else:
            interesting_members = []
            mediocre_members = []
        
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
                        interesting_members.append(optim_node)
                        mediocre_members = []
            return (interesting_members, mediocre_members)

        min_gmrc = new_gmrc.copy()
        min_gmrc.modify_grsp_ang(grip, -GMRC.grsp_ang_cap)
        min_ang = min_gmrc.get_grip_gamma(grip)
        if min_ang < mid_ang:
            min_node = TreeNode(min_gmrc, self.cgf_manager.copy(),
                                self, self.g_depth + 1, self.mediocrity + 1, self.tree)
            if min_node.is_novel:
                if self.mediocrity < TreeNode.mediocrity_tolerance:
                    interesting_members.append(min_node)
                else:
                    mediocre_members.append(min_node)

        max_gmrc = new_gmrc.copy()
        max_gmrc.modify_grsp_ang(grip, GMRC.grsp_ang_cap)
        max_ang = max_gmrc.get_grip_gamma(grip)
        if max_ang > mid_ang:
            max_node = TreeNode(max_gmrc, self.cgf_manager.copy(),
                                self, self.g_depth + 1, self.mediocrity + 1, self.tree)
            if max_node.is_novel:
                if self.mediocrity < TreeNode.mediocrity_tolerance:
                    interesting_members.append(max_node)
                else:
                    mediocre_members.append(max_node)

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
                ang = choice[0]
                if ang <= min_ang and ang >= max_ang:
                    continue
                ss_gmrc = new_gmrc.copy()
                if not ss_gmrc.modify_grsp_ang(grip, ang):
                    continue
                if np.abs(ss_gmrc.get_grip_gamma(grip) - ang) > 1e-3:
                    continue
                ss_node = TreeNode(ss_gmrc, ss_cgf_manager, 
                                   self, self.g_depth + 1, 0, self.tree)
                if ss_node.is_novel:
                    interesting_members.append(ss_node)
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
                bc_node = TreeNode(bc_gmrc, bc_cgf_manager, 
                                self, self.g_depth + 1, 0, self.tree)
                if bc_node.is_novel:
                    interesting_members.append(bc_node)
                if bc_node.is_goal():
                    break

        return (interesting_members, mediocre_members)

class Tree:
    def __init__(self, gmrc: GMRC, cgf_manager: CGFManager, tar_gmrc: GMRC,
                 ed_estimator: EDEstimator, id_verdict: IDVerdict):
        self.nodes_at_depth: list[TreeNode] = [[]]
        self.ethinicity2id = dict()

        self.max_g_depth = 0
        self.target_gmrc = tar_gmrc

        self.ed_estimator = ed_estimator
        self.id_verdict = id_verdict

        self.root = TreeNode(gmrc,
                        cgf_manager=cgf_manager,
                        parent=None,
                        g_depth=0,
                        mediocrity=0,
                        tree = self)

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
    """
    def plan(self, gmrc_1: GMRC, gmrc_2: GMRC, method = "BFS"):
        GMRC.suppress_action_err = True
        assert gmrc_1.m == gmrc_2.m
        CGFManager.m = gmrc_1.m
        cgf_manager = CGFManager(gmrc_2.get_Gamma_final())
        self.tree = Tree(gmrc_1, cgf_manager, gmrc_2, self.ed_estimator, self.id_verdict)

        if method == "BFS":
            while True:
                node = self.tree.push_front()
                if node is not None:
                    break
        elif method == "DMT_BFS":
            node = self.tree.explore()

        path = [node]
        while node.parent is not None:
            node = node.parent
            path.append(node)
        path.reverse()
        return path