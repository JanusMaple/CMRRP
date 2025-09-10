"""
Constraint Manifold for Modular Robot Configuration Motion Planning
"""

import sys
sys.path.append('..')
import torch
torch.pi = torch.acos(torch.zeros(1)).item() * 2
from GMRC import GMRC

class CMMRC:
    """
    Extracts key parameters for representing the constraint manifold

    Parameters: 
    ----------
    gmrc_1: GMRC
        The GMRC before a grasping action
    
    gmrc_2: GMRC
        The GMRC after a grasping action
    """
    def __init__(self, gmrc_1: GMRC, gmrc_2: GMRC, device = None):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.n = gmrc_1.m                           # Dimension of the space
        self.avatar = gmrc_1.copy()                 # Avatar GMRC for collision check
        self.cyc_num = gmrc_1.c                     # Number of constraints
        self.ba_cyc_loops = [torch.tensor(module_loop, 
                                        dtype=torch.long, 
                                        device=self.device) 
                           for module_loop in gmrc_1.module_loops]
        self.bas_loops = [torch.tensor(module_ht_loop, 
                                       dtype=torch.long, 
                                       device=self.device) 
                          for module_ht_loop in gmrc_1.module_ht_loops]
        self.ang_sum_tars = torch.pi * 2 * \
            torch.tensor(gmrc_1.loop_polarities, device=self.device)
        self.gammas_list = []
        for i in range(self.cyc_num):
            self.gammas_list.append(
                torch.from_numpy(
                    gmrc_2.grsp_angs[gmrc_2.ga_gi_loops[i]] * gmrc_2.gas_loops[i]
                ).to(self.device)
            )
        
        self.two_cyc_tars = [None] * self.cyc_num
        self.is_2_cyc = [False] * self.cyc_num
        self.constraint_num = 0
        bend_angs = torch.tensor(gmrc_1.bend_angs, device=self.device)
        for cyc in range(self.cyc_num):
            if gmrc_1.is_2_cycle(cyc):
                self.constraint_num = self.constraint_num + 2
                self.is_2_cyc[cyc] = True
                betas = bend_angs[self.ba_cyc_loops[cyc]] * self.bas_loops[cyc]
                self.two_cyc_tars[cyc] = betas
            else:
                self.constraint_num = self.constraint_num + 3

    def constraint_func(self, x: torch.tensor):
        error = torch.zeros(self.constraint_num, 
                            dtype=torch.float, 
                            device=self.device)
        i = 0
        cyc = 0
        while i < self.constraint_num:
            betas = x[self.ba_cyc_loops[cyc]] * self.bas_loops[cyc]
            if self.is_2_cyc[cyc]:
                error[i] = torch.abs(self.two_cyc_tars[cyc][0] - betas[0])
                error[i + 1] = torch.abs(self.two_cyc_tars[cyc][1] - betas[1])
                i = i + 2
            else:
                gammas = self.gammas_list[cyc]
                ang_sum_tar = self.ang_sum_tars[cyc]
                loop_length = len(self.ba_cyc_loops[cyc])
                loop_angle_error = torch.abs(
                    torch.sum(betas) + torch.sum(gammas) - ang_sum_tar
                    )
                dx, dy = self.get_single_loop_dock_error(betas, gammas, loop_length)
                error[i] = loop_angle_error
                error[i + 1] = dx
                error[i + 2] = dy
                i = i + 3
            cyc = cyc + 1
            
        return error

    def collision_func(self, x: torch.tensor):
        in_boundary = (x >= -GMRC.mdl_ang_cap) & (x <= GMRC.mdl_ang_cap)
        if not in_boundary.any():
            return True
        self.avatar.bend_angs = x.detach().cpu().numpy()
        self.avatar.update_all_module_geometry()
        self.avatar.update_all_module_collider()
        return self.avatar.is_collision_detected()
    
    @staticmethod
    def get_single_loop_dock_error(betas: torch.Tensor, 
                                   gammas: torch.Tensor, 
                                   l: int) -> torch.Tensor:
        device = betas.device

        a = torch.zeros((l, 1), device=device)
        b = torch.zeros((l, 1), device=device)

        for j in range(GMRC.num_seg_lens):
            cur_betas = betas / (GMRC.num_seg_lens - 1) * j
            a = a + torch.cos(cur_betas) * GMRC.mdl_seg_lens[j]
            b = b + torch.sin(cur_betas) * GMRC.mdl_seg_lens[j]

        beta_cml = torch.cumsum(betas, dim=0)
        zero = torch.tensor([0.0], device=device)
        beta_cml = torch.cat([zero, beta_cml[0:l - 1]])

        gamma_cml = torch.cumsum(gammas, dim=0) - gammas[0]
        alphas = beta_cml + gamma_cml

        a = a.squeeze(-1)
        b = b.squeeze(-1)

        delta_x = torch.sum(a * torch.cos(alphas) - b * torch.sin(alphas))
        delta_y = torch.sum(a * torch.sin(alphas) + b * torch.cos(alphas))

        return delta_x, delta_y
