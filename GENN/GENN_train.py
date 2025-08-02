from tqdm import tqdm

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.data import DataLoader

from GENN import GENN, DegreeEmbedding, SequentialPooling
from GENN_data import GENNDataset

parser = argparse.ArgumentParser(description="Select data generation mode: RW/BFS")
parser.add_argument('--mode', type=str, default="rw",
                    help='Generation Mode: rw or bfs')
args = parser.parse_args()
if args.mode == "rw":
    print("Generating data with inaccurate distance using random walk")
    root = "random_walk"
    model_path = "model/rw_trained_model.pth"
elif args.mode == "bfs":
    print("Generating data with exact distance using breadth first search")
    root = "breadth_first_search"
    model_path = "model/bfs_trained_model.pth"
else:
    raise ValueError("\033[91mWrong mode for training model: use rw or bfs\033[0m")

"""
Traning Parameters
"""
max_epoch_num = 100
early_stop_threshold = 0.005
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = GENNDataset(emrc_pairs=None, is_test=False, root=root)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

degree_embedding = DegreeEmbedding(embed_dim=16, device=device)
gnn = GENN(16, 32, 32, device)
pooling = SequentialPooling(32, 32, 16, device)

optimizer = torch.optim.Adam(
    list(degree_embedding.parameters()) +
    list(gnn.parameters()) +
    list(pooling.parameters()),
    lr=1e-3, weight_decay=1e-5)

last_loss = None
largest_error = None
for epoch in range(max_epoch_num):
    gnn.train()
    pooling.train()
    degree_embedding.train()
    total_loss = 0
    for batch in tqdm(loader, desc=f"Epoch {epoch+1}", leave=False):
        optimizer.zero_grad()

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

        graph_feat_diff = graph_feat_1 - graph_feat_2
        predicted_distance = graph_feat_diff.norm(p=2, dim=-1)

        loss = F.mse_loss(predicted_distance, distances)
        loss.backward()
        optimizer.step()

        total_loss = total_loss + loss.item()

    print(f"Epoch {epoch+1} | Loss: {total_loss:.4f}")
    if last_loss is not None:
        if largest_error is not None:
            new_loss = last_loss - total_loss
            if new_loss / largest_error < early_stop_threshold:
                break
        else:
            largest_error = last_loss - total_loss
    last_loss = total_loss

torch.save({
    'gnn': gnn.state_dict(),
    'degree_embedding': degree_embedding.state_dict(),
    'pooling': pooling.state_dict()
}, model_path)
