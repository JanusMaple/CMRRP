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
    def get_random_configuration(seed = None, w = None, v = None):
        # Get the total number of modules
        m = MRC.n_modules

        modules = []
        for i in range(m):
            modules.append(Module())

        gmrc = GMRC.get_random_configuration(m, seed, w, v)
        
        # Grasp process to form an MRC
        Module.start_hacking()
        idx_g = 0
        while idx_g < len(gmrc.grippers):
            # If is the first gripper in grip or second gripper in w grip
            if (idx_g % 3 == 0) or ((idx_g % 3 == 1) and (gmrc.grippers[idx_g + 1] > -2)):
                idx_m_catcher = gmrc.gripper2module[idx_g] // 2
                ht_catcher = gmrc.gripper2module[idx_g] % 2
                idx_m_catched = gmrc.gripper2module[idx_g + 1] // 2
                ht_catched = gmrc.gripper2module[idx_g + 1] % 2
                grsp_theta = GraspTheta(gmrc.grsp_angs[idx_g])
                modules[idx_m_catcher].catch(
                    modules[idx_m_catched], ht_catched, ht_catcher, grsp_theta)

                if idx_g % 3 == 0:
                    idx_g = idx_g + 1
                else:
                    idx_g = idx_g + 2
            elif (idx_g % 3 == 1) and (gmrc.grippers[idx_g + 1] == -2):
                idx_g = idx_g + 2
            else:
                idx_g = idx_g + 1           # Should never happen though
        for idx_m in range(m):
            modules[idx_m].bend_to(gmrc.bend_angs[idx_m])
        Module.stop_hacking()

        return MRC(modules, gmrc)
