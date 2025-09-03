import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch_geometric.nn import MessagePassing

class GGNN(MessagePassing):
    max_num_degree = 3

    """
    Sequence-Sensitive Graph Neural Network for Graph Embedding

    Parameters
    ----------
    in_dim: int
        input feature dimension for each node

    out_dim: int
        output feature dimension for each node

    hidden_dim: int
        hidden feature dimension of sequence model

    device: torch.device
        the device where this model is implemented
    """
    def __init__(self, in_dim, out_dim, hidden_dim, device=None):
        super().__init__(aggr='add',                    # Sum
                         node_dim=0)
        self.seq_model = nn.RNN(
            in_dim + 1,                                 # One extra edge feature (phi)
            hidden_dim,
            num_layers=2,
            batch_first=False,
            bidirectional=False,                        # NOTE: Always False for GGNN
            device=device
            )
        self.update_mlp = nn.Linear(
            in_dim + hidden_dim,
            out_dim,
            device=device
            )
        self.batch_norm = nn.BatchNorm1d(
            out_dim, 
            device=device
        )

    def reset_parameters(self):
        return super().reset_parameters()

    """
        Model forward function for GNN message passing with sequence model

        Parameters
        ----------
        x: torch.tensor
            size: [num_nodes, in_dim]; dtype: torch.float
        edge_index: torch.tensor
            size: [2, num_edges]; dtype: torch.long
        cyclic_neighbors: torch.tensor
            size: [num_nodes, max_num_degree = 3]; dtype: torch.long
        neighbor_phis: torch.tensor
            size: [num_nodes, max_num_degree = 3]; dtype: torch.float
        neighbor_num: torch.tensor
            size: [num_nodes, ]; dtype: torch.long
        """
    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                cyclic_neighbors: torch.Tensor,
                neighbor_phis: torch.Tensor,
                neighbor_num: torch.Tensor):
        node_num = x.size()[0]
        hidden_dim = self.seq_model.hidden_size

        neighbor_feats = []
        lengths_list = []
        for i in range(neighbor_num.size()[0]):
            neighbors_i = cyclic_neighbors[i, 0 : neighbor_num[i]]
            phis_i = neighbor_phis[i, 0 : neighbor_num[i]]
            for j in range(neighbor_num[i]):
                neighbor_feat = x[neighbors_i.roll(j, dims = 0)]
                neighbor_feats.append(
                    torch.cat(
                        [neighbor_feat, 
                         phis_i.roll(j, dims = 0).unsqueeze(1)], 
                        dim=-1)
                    )
                lengths_list.append(neighbor_num[i])
        padded_feats = pad_sequence(neighbor_feats, batch_first=False)
        lengths = torch.tensor(lengths_list, dtype=torch.long).cpu()
        packed_feats = pack_padded_sequence(padded_feats, lengths=lengths, 
                                            batch_first=False, enforce_sorted=False)
        _, h_n = self.seq_model(packed_feats)
        seq_out = torch.zeros(node_num, hidden_dim)     # size: (node_num, hidden_dim)
        seq_out = seq_out.to(x.device)

        index = 0
        for i in range(node_num):                       # Sample anchored sequences
            seq_out[i, :] = h_n[-1, index : index + neighbor_num[i], :].mean(dim=0)
            index = index + neighbor_num[i]

        neighbor_num = neighbor_num.float()
        return self.propagate(edge_index, x=x, seq=seq_out, neighbor_num=neighbor_num)

    """
    Message passing from node j to node i
    """
    def message(self, seq_i, neighbor_num_i):           # foo_i = foo[edge_index[0]]
        return seq_i / neighbor_num_i.unsqueeze(-1)     # Normalization

    """
    Update aggregation output with original node feature

    Parameters
    ----------
    aggr_out: torch.tensor; size: (node_num, hidden_dim)

    x: torch.tensor; size: (node_num, input_dim)
    """
    def update(self, aggr_out, x):
        out = self.update_mlp(torch.cat([x, aggr_out], dim=-1))
        out = self.batch_norm(out)
        out = F.tanh(out)
        return out

class DegreeEmbedding(nn.Module):
    """
    Embed original one hot vectors from node derees into a higher dimension
    """
    def __init__(self, embed_dim, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(GGNN.max_num_degree, embed_dim, device=device),
            nn.Tanh(),
            nn.Linear(embed_dim, embed_dim, device=device)
        )

    def forward(self, x):
        return self.mlp(x)
    
class SequentialPooling(nn.Module):
    """
    Pooling model for converting node level features to grap level features
    """

    def __init__(self, in_dim, hidden_dim, out_dim, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seq_model = nn.RNN(
            input_size=in_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=False,
            bidirectional=False,
            device=device
        )

        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, device=device),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim, device=device)
        )

    def forward(self, x, node_count):
        # node_count: number of nodes in each graph within this batch
        x_mean = x.mean(dim=-1)
        graph_feats = []
        j = 0
        for i in range(len(node_count)):
            graph_feats.append(x[x_mean[j : j + node_count[i]].argsort() + j, :])
            j = j + node_count[i]
        padded_feats = pad_sequence(graph_feats, batch_first=False)
        packed_feats = pack_padded_sequence(padded_feats, node_count.cpu(), 
                                            batch_first=False, enforce_sorted=False)
        _, h_n = self.seq_model(packed_feats)
        return self.mlp(h_n[-1])
