from Grasp import Grasp
from GraspTheta import GraspTheta

class Module:
    _hack_mode = False

    @staticmethod
    def start_hacking():
        Module._hack_mode = True

    @staticmethod
    def stop_hacking():
        Module._hack_mode = False

    def __init__(self):
        self.grasps = [Grasp(), Grasp()]    # The grasp status of head and tail grippers
        self.beta = 0
        self.gamma = 0
        self.name = "M0"
        self.index = 0

    def get_index(self):
        return self.index

    def set_index(self, index):
        self.index = index

    def rename(self, name):
        self.name = name

    def get_name(self):
        return self.name
    
    def get_description(self):
        output = ""
        if self.grasps[0].is_module_in_hand():
            output = self.name + " head" + self.grasps[0].get_grasp_description()
            if self.grasps[1].is_module_in_hand():
                output = output + "\n"
        if self.grasps[1].is_module_in_hand():
            output = output + self.name + " tail" + self.grasps[1].get_grasp_description()
        return output

    # Whether does the head or tail gripper is grasping
    def is_hand_full(self, head_tail):
        return self.grasps[head_tail].is_module_in_hand()

    # Use self's head_tail to capture module's tar_head_tail
    def catch(self, module, tar_head_tail, head_tail, theta = GraspTheta()):
        assert isinstance(module, Module)
        if not Module._hack_mode:       # Hack mode can skip all the check, ignore rules
            if not self.grasps[head_tail].can_catch():
                return False
            if not module.can_be_catched(tar_head_tail):
                return False

        self.grasps[head_tail].catch(module, tar_head_tail, theta)
        module.be_catched_by(self, tar_head_tail, head_tail, theta)

        return True
    
    # Bend to a specific angle
    def bend_to(self, tar_beta = 0, tar_gamma = 0):
        self.beta = tar_beta
        self.gamma = tar_gamma
    
    # Use module's tar_head_tail to capture self's head_tail
    def be_catched_by(self, module, tar_head_tail, head_tail, theta = GraspTheta()):
        self.grasps[head_tail].be_catched_by(module, tar_head_tail, theta)

    # head_tail is either 0 or 1, for head or tail
    def release(self, head_tail):
        if not Module._hack_mode:       # Hack mode can skip all the check, ignore rules
            if not self.grasps[head_tail].can_release():
                return False
        self.grasps[head_tail].release()

        return True
    
    # The head_tail gripper will hence be released
    def be_released(self, head_tail):
        self.grasps[head_tail].be_released()
    
    def can_be_catched(self, head_tail):
        return self.grasps[head_tail].can_be_catched()
    
    def get_neighbors(self):
        neighbors = []
        for i in range(2):
            neighbors.extend(self.grasps[i].get_neighbors())
        return neighbors