"""
Continuum Modular Robot Reconfigure Planner: Task and Motion Planning
"""

from __future__ import annotations
import sys
sys.path.append('..')
sys.path.append('../GENN')
import copy
from GMRC import GMRC
from GENN import GENN, DegreeEmbedding, SequentialPooling

# The manager for correspondence and Gamma_final
class CGFManager:
    m: int = 7                              # Number of modules in a configuration

    # The initialization methods should only be called for the root node, once
    def __init__(self, Gamma_final: list, correspondence: list = None, 
                 gf_ang_list: list = None, gt_ang_list: list = None):
        self.Gamma_final = Gamma_final

        # correspondence[2 * i + 0] = 2 * j + 1 => i's head is j's tail in target gmrc
        if correspondence is None:
            self.correspondence = [-1] * (2 * CGFManager.m)
        else:
            self.correspondence = correspondence

        # Lists storing the appropriate angle *if* gf is grasping gt
        #   NOTE: Which means that the number of grips here is redundant
        #         For example, 3H==30°==>6T and 6T==-30°==>3H both exist
        if gf_ang_list is None:             # gf_ang_list[2 * i + ht] = (ang, gt)
            gf_ang_list = [None] * (2 * CGFManager.m)
            for gamma in self.Gamma_final:
                pass
        else:
            self.gf_ang_list = gf_ang_list
        if gt_ang_list is None:             # gt_ang_list[2 * i + ht] = (ang, gf)
            gt_ang_list = [None] * (2 * CGFManager.m)
        else:
            self.gt_ang_list = gt_ang_list

    # TODO: If gt's correspondence is not established and gt is on w-grip layer-2
    #       Then return None, indicating that should not allow choosing from Gamma_final
    def get_Gamma_final_angle(self, gf, gt):    # Get the appropriate Gamma_final angle
        pass

    def copy(self):                         # Corresponding will not change ang_lists
        return CGFManager(copy.deepcopy(self.Gamma_final),
                          copy.deepcopy(self.correspondence),
                          self.gf_ang_list, self.gt_ang_list)

# A search tree node contains: 1. a unique gmrc shape; 2. a partial correspondence
class TreeNode:
    def __init__(self, gmrc: GMRC, cgf_manager: CGFManager, 
                 parent: TreeNode = None, g_depth: int = 0):
        self.cgf_manager = cgf_manager
        self.gmrc: GMRC = gmrc
        self.parent: TreeNode = parent
        self.g_depth: int = g_depth
        self.children: list = []
        self.mediocrity: int = 0                            # TODO: Make this useful
        self.expanded = False
        
    def expand(self, tar_g_depth = None):
        if tar_g_depth is not None:
            if self.g_depth >= tar_g_depth:
                return
        actions = self.gmrc.get_all_actions()
        for action in actions:
            new_gmrc = self.gmrc.copy()
            if action is not tuple:                         # Release
                new_gmrc.execute_action(action)
                new_node = TreeNode(new_gmrc, self.cgf_manager.copy(),
                                    self, self.g_depth)
                self.children.append(new_node)
            else:                                           # Grasp
                if not new_gmrc.execute_action(action):
                    continue
                self.children.extend(
                    self._get_all_memebers_in_grasping_group(new_gmrc, action))

        self.expanded = True

    # Get all reasonable children from the same grasping action based on eldest sibling
    # NOTE: Will not have a->b with alpha and b->a with alpha
    #       since will only have a->b from EMRC.get_all_actions()
    def _get_all_memebers_in_grasping_group(self, new_gmrc: GMRC, action: tuple):
        optim_node = TreeNode(new_gmrc, self.cgf_manager.copy(),
                              self, self.g_depth + 1)
        members = [optim_node]
        if len(new_gmrc.module_loops[-1]) <= 2:
            return members
        gf = action[0]
        gt = action[1]
        grip = new_gmrc.module2gripper[gf % 2][gf // 2] // 3
        all_Gamma_final_angs = self.cgf_manager.get_Gamma_final_angle(gf, gt)
        return members

class CMRRP:
    def __init__(self, g: GENN, d: DegreeEmbedding, s: SequentialPooling):
        self.genn = g
        self.degree_embedding = d
        self.sequential_pooling = s

    def plan(self, gmrc_1: GMRC, gmrc_2: GMRC):
        assert gmrc_1.m == gmrc_2.m
        CGFManager.m = gmrc_1.m
        root = TreeNode(gmrc_1,
                        cgf_manager=CGFManager(gmrc_2.get_Gamma_final()),
                        parent=None,
                        g_depth=0)
