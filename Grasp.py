from GraspTheta import GraspTheta

# The status of a gripper
class Grasp:
    ht_index_2_string = [" head", " tail"]

    def __init__(self):
        self.module_in_hand = False
        self.min_hand_theta = GraspTheta()
        self.min_hand_module = None
        self.min_hand_head_tail = 0   # 0 is head; 1 is tail
        
        self.module_out_hand = False
        self.mout_hand_theta = GraspTheta()
        self.mout_hand_module = None
        self.mout_hand_head_tail = 0   # 0 is head; 1 is tail

    def get_neighbors(self):
        neighbors = []
        if self.module_in_hand:
            neighbors.append(self.min_hand_module.get_index())
        if self.module_out_hand:
            neighbors.append(self.mout_hand_module.get_index())
        return neighbors

    def get_grasp_description(self):
        description = ""
        if self.module_in_hand:
            description = " holds " + self.min_hand_module.get_name() + Grasp.ht_index_2_string[self.min_hand_head_tail]
        return description
    
    def catch(self, module, head_tail, theta = GraspTheta()):
        assert isinstance(theta, GraspTheta)

        self.module_in_hand = True
        self.min_hand_module = module
        self.min_hand_theta = theta
        self.min_hand_head_tail = head_tail

    def be_catched_by(self, module, head_tail, theta = GraspTheta()):
        assert isinstance(theta, GraspTheta)

        self.module_out_hand = True
        self.mout_hand_module = module
        self.mout_hand_theta = theta
        self.mout_hand_head_tail = head_tail

    def release(self):
        self.module_in_hand = False
        self.min_hand_module.be_released(self.min_hand_head_tail)

    def be_released(self):
        self.module_out_hand = False
    
    def is_module_in_hand(self):        # Whether is there module in hand
        return self.module_in_hand

    def can_catch(self):
        if self.module_in_hand or self.module_out_hand:
            return False
        return True
    
    def can_be_catched(self):
        if self.module_out_hand:
            return False
        if self.module_in_hand:
            if self.min_hand_module.is_hand_full(self.min_hand_head_tail):
                return False
        return True
    
    def can_release(self):
        if self.module_in_hand and not self.module_out_hand:
            return True
        return False