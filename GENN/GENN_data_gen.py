"""
Generate data for training GENN
"""

import sys
from tqdm import tqdm
import torch
import numpy as np

sys.path.append('..')
from EMRC import EMRC
from GENN_data import GENNDataset

module_number = 7
rw_steps = [0, 1, 4, 9, 16, 25, 36]
data_size_per_step = 1024

emrc_pairs = []

for i in tqdm(range(data_size_per_step * len(rw_steps))):
    rw_step = rw_steps[i % len(rw_steps)]
    emrc_1 = EMRC.get_random_configuration(module_number)
    emrc_2 = EMRC(w=emrc_1.w,
                  v=emrc_1.v,
                  n=emrc_1.n,
                  m=emrc_1.m,
                  grippers=emrc_1.grippers,
                  gripper2module=emrc_1.gripper2module,
                  module2gripper=emrc_1.module2gripper,
                  rng=np.random.Generator(emrc_1.rng.bit_generator),
                  loop_polarities=emrc_1.loop_polarities,
                  grip_polarities=emrc_1.grip_polarities)
    for j in range(rw_step):
        emrc_2.execute_random_action()
    distance = torch.tensor([np.sqrt(rw_step)], dtype=torch.float)
    emrc_pairs.append((emrc_1, emrc_2, distance))

dataset = GENNDataset(emrc_pairs)
