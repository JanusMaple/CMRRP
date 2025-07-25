import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch_geometric.nn import MessagePassing

class GENN(MessagePassing):
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
        super().__init__(aggr='add')                    # Sum
        self.seq_model = nn.RNN(
            in_dim,
            hidden_dim,
            num_layers=2,
            batch_first=False,
            bidirectional=False,                        # NOTE: Always False for GENN
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
        neighbor_num: torch.tensor
            size: [num_nodes, ]; dtype: torch.long
        """
    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor, 
                cyclic_neighbors: torch.Tensor, 
                neighbor_num: torch.Tensor):
        node_num = x.size()[0]
        hidden_dim = self.seq_model.hidden_size

        neighbor_feats = []
        for i in range(neighbor_num.size()[0]):
            nieghbors_i = cyclic_neighbors[i, 0 : neighbor_num[i]]
            for j in range(neighbor_num[i]):
                neighbor_feat = x[nieghbors_i.roll(j, dim = 0)]
                neighbor_feats.append(neighbor_feat)
        padded_feats = pad_sequence(neighbor_feats, batch_first=False)
        packed_feats = pack_padded_sequence(padded_feats, neighbor_num, 
                                            batch_first=False, enforce_sorted=False)
        _, h_n = self.seq_model(packed_feats)
        seq_out = torch.zeros(node_num, hidden_dim)     # size: (node_num, hidden_dim)
        
        index = 0
        for i in range(node_num):                       # Sample anchored sequences
            seq_out[i, :] = h_n[-1, index : index + neighbor_num[i], :].mean(dim=0)
            index = index + neighbor_num[i]

        return self.propagate(edge_index, x=x, seq=seq_out, neighbor_num=neighbor_num)

    """
    Message passing from node j to node i
    """
    def message(self, seq_i, neighbor_num_i):           # foo_i = foo[i]
        return seq_i / neighbor_num_i                   # Normalization

    """
    Update aggregation output with original node feature

    Parameters
    ----------
    aggr_out: torch.tensor; size: (node_num, hidden_dim)

    x: torch.tensor; size: (node_num, input_dim)
    """
    def update(self, aggr_out, x):
        out = self.update_mlp(torch.cat([x, aggr_out], dim=-1))
        out_norm = self.batch_norm(out)
        return out_norm

class DegreeEmbedding(nn.Module):
    """
    Embed original one hot vectors from node derees into a higher dimension
    """
    def __init__(self, embed_dim, device=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mlp = nn.Sequential(
            nn.Linear(GENN.max_num_degree, embed_dim, device=device),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim, device=device)
        )

    def forward(self, x):
        return self.mlp(x)
