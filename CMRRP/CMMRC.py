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
        self.constraint_num = gmrc_1.c              # Number of constraints
        self.ba_i_loops = [torch.tensor(module_loop, 
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
        for i in range(self.constraint_num):
            self.gammas_list.append(
                torch.from_numpy(
                    gmrc_2.grsp_angs[gmrc_2.ga_gi_loops[i]] * gmrc_2.gas_loops[i]
                ).to(self.device)
            )

    def constraint_func(self, x: torch.tensor):
        error = torch.zeros(2 * self.constraint_num, 
                            dtype=torch.float, 
                            device=self.device)
        for i in range(self.constraint_num):
            betas = x[self.ba_i_loops[i]] * self.bas_loops[i]
            gammas = self.gammas_list[i]
            ang_sum_tar = self.ang_sum_tars[i]
            loop_length = len(self.ba_i_loops[i])
            loop_angle_error = torch.abs(
                torch.sum(betas) + torch.sum(gammas) - ang_sum_tar
                )
            loop_dock_error = self.get_single_loop_dock_error(betas, gammas, loop_length)
            error[2 * i] = loop_angle_error
            error[2 * i + 1] = loop_dock_error
            
        return error

    def is_rejecting(self, x: torch.tensor):
        in_boundary = (x >= -GMRC.mdl_ang_cap) & (x <= GMRC.mdl_ang_cap)
        if not in_boundary.any():
            return True
        self.avatar.bend_angs = x.cpu().numpy()
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

        return torch.sqrt(delta_x ** 2 + delta_y ** 2)
