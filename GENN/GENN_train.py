import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_mean_pool

from GENN import GENN, DegreeEmbedding, SequentialPooling
from GENN_data import GENNDataset

"""
Traning Parameters
"""
epoch_num = 1
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = GENNDataset(None)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

degree_embedding = DegreeEmbedding(embed_dim=16, device=device)
gnn = GENN(16, 32, 32, device)
pooling = SequentialPooling(32, 16, device)

for epoch in range(epoch_num):
    for batch in loader:
        x_1 = batch.x_dict['emrc_1'].to(device)
        edge_index_1 = batch.edge_index_dict[
            ('emrc_1', 'emrc_1_edges', 'emrc_1')].to(device)
        cyclic_neighbors_1 = batch['emrc_1'].cyclic_neighbors.to(device)
        neighbor_num_1 = batch['emrc_1'].neighbor_num.to(device)

        x_2 = batch.x_dict['emrc_2'].to(device)
        edge_index_2 = batch.edge_index_dict[
            ('emrc_2', 'emrc_2_edges', 'emrc_2')].to(device)
        cyclic_neighbors_2 = batch['emrc_2'].cyclic_neighbors.to(device)
        neighbor_num_2 = batch['emrc_2'].neighbor_num.to(device)

        distances = batch.distance.to(device)

        x_oh_1 = F.one_hot(x_1, GENN.max_num_degree).float()
        x_oh_2 = F.one_hot(x_2, GENN.max_num_degree).float()
        
        x_degree_feat_1 = degree_embedding(x_oh_1)
        x_degree_feat_2 = degree_embedding(x_oh_2)

        x_gnnout_feat_1 = gnn(
            x_degree_feat_1, edge_index_1, cyclic_neighbors_1, neighbor_num_1)
        x_gnnout_feat_2 = gnn(
            x_degree_feat_2, edge_index_2, cyclic_neighbors_2, neighbor_num_2)

        graph_feat_1 = pooling(x_gnnout_feat_1, torch.bincount(batch['emrc_1'].batch))
        graph_feat_2 = pooling(x_gnnout_feat_2, torch.bincount(batch['emrc_2'].batch))
        break
    break
