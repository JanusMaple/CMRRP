from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from GENN import GENN, DegreeEmbedding, SequentialPooling
from GENN_data import GENNDataset

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

"""
Test Model
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = GENNDataset(emrc_pairs=None, is_test=True)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

degree_embedding = DegreeEmbedding(embed_dim=16, device=device)
gnn = GENN(16, 32, 32, device)
pooling = SequentialPooling(32, 32, 16, device)

checkpoint = torch.load("model/model_checkpoint.pth", map_location=device)
gnn.load_state_dict(checkpoint['gnn'])
pooling.load_state_dict(checkpoint['pooling'])
degree_embedding.load_state_dict(checkpoint['degree_embedding'])

gnn.eval()
pooling.eval()
degree_embedding.eval()

dis_label_num = 7

grouped_pre_dis = [torch.tensor([], dtype=torch.float).to(device)] * dis_label_num

for batch in tqdm(loader):
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

    distances = batch.distance.to(device).to(torch.long)

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

    graph_feat_diff = graph_feat_1 - graph_feat_2
    predicted_distance = graph_feat_diff.norm(p=2, dim=-1)
    
    grouped_pre_dis_b = [predicted_distance[distances == i] for i in range(7)]

    for i in range(len(grouped_pre_dis)):
        grouped_pre_dis[i] = torch.cat((grouped_pre_dis[i], grouped_pre_dis_b[i]))

for i in range(len(grouped_pre_dis)):
    grouped_pre_dis[i] = grouped_pre_dis[i].cpu().detach().numpy()

group_len = len(grouped_pre_dis[0])
group_labels = []
for i in range(dis_label_num):
    group_labels = group_labels + [f"{i}"] * group_len

df = pd.DataFrame({
    'value': np.concatenate(grouped_pre_dis),
    'group': group_labels
})

sns.boxplot(x='group', y='value', data=df)
plt.title("Model Test Results")
plt.xlabel("Random Walk Distance")
plt.ylabel("Model Predicted Distance")
plt.show()
