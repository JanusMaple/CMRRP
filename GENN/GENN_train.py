import torch
from torch_geometric.data import DataLoader

from GENN_data import GENNDataset

"""
Traning Parameters
"""
epoch_num = 1
batch_num = 32

dataset = GENNDataset(None)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(epoch_num):
    for batch in loader:
        print(batch.batch_size)
