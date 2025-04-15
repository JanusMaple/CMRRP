# Three floats representing the grasping angle

import numpy as np

class GraspTheta:
    def __init__(self, pitch = 0.0, roll = 0.0, yaw = 0.0):
        self.pitch = np.arctan2(np.sin(pitch), np.cos(pitch))
        self.roll = np.arctan2(np.sin(roll), np.cos(roll))
        self.yaw = np.arctan2(np.sin(yaw), np.cos(yaw))

    # NOTE: Should be Deprecated
    def is_legal(self):
        if np.abs(self.pitch) > np.pi / 3:
            return False
        else:
            return True
    
    # NOTE: Should be Deprecated
    @staticmethod
    def get_rnd_legal(rng):
        assert isinstance(rng, np.random.Generator)
        pitch = rng.uniform(-np.pi / 3, np.pi / 3)
        roll = 0
        yaw = 0
        return GraspTheta(roll, pitch, yaw)