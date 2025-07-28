"""
Generate data for training GENN
"""

import argparse
import sys
from tqdm import tqdm
import torch
import numpy as np

parser = argparse.ArgumentParser(description="Select data generation mode: RW/BFS")
parser.add_argument('--mode', type=str, default="rw",
                    help='Generation Mode: rw or bfs')
args = parser.parse_args()
if args.mode == "rw":
    print("Generating data with inaccurate distance using random walk")
    root = "random_walk"
elif args.mode == "bfs":
    print("Generating data with exact distance using breadth first search")
    root = "breadth_first_search"
else:
    raise ValueError("\033[91mWrong mode for generating data: use rw or bfs\033[0m")

sys.path.append('..')
sys.path.append('../GENN')
from EMRC import EMRC
from GENN_data import GENNDataset
from GENN import GENN, DegreeEmbedding, SequentialPooling

if args.mode == "bfs":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    degree_embedding = DegreeEmbedding(embed_dim=16, device=device)
    gnn = GENN(16, 32, 32, device)
    pooling = SequentialPooling(32, 32, 16, device)
    checkpoint = torch.load("model/model_checkpoint.pth", map_location=device)
    gnn.load_state_dict(checkpoint['gnn'])
    pooling.load_state_dict(checkpoint['pooling'])
    degree_embedding.load_state_dict(checkpoint['degree_embedding'])

rng = np.random.default_rng()
seed_bias = rng.integers(100000, 1000000)
print(f"\033[94mseed_bias is: {seed_bias}\033[0m")

module_number = 7
distances = [0, 1, 2, 3, 4, 5, 6]
data_size_per_step = 32768

emrc_pairs = []

for i in tqdm(range(data_size_per_step * len(distances))):
    emrc_1 = EMRC.get_random_configuration(module_number, seed=i + seed_bias)
    if args.mode == "rw":
        emrc_2 = emrc_1.copy()
        rw_step = distances[i % len(distances)] * distances[i % len(distances)]
        for j in range(rw_step):
            emrc_2.execute_random_action()
        distance = torch.tensor([distances[i % len(distances)]], dtype=torch.float)
    elif args.mode == "bfs":
        pass
    emrc_pairs.append((emrc_1, emrc_2, distance))

dataset = GENNDataset(emrc_pairs=emrc_pairs, is_test=False, root=root, force_reload=True)