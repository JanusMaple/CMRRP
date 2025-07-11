import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

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
        super().__init__(aggr='None')                   # By None it uses aggregate()
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

    def reset_parameters(self):
        return super().reset_parameters()

    def forward(self, x: torch.Tensor, 
                edge_index: torch.Tensor, 
                cyclic_neighbors: list):
        # x: [num_nodes, in_dim], dtype: float
        # edge_index: [2, num_edges], dtype: long
        # cyclic_neighbors: [torch.tensor(dtype = torch.long)] * num_nodes

        neighbor_feats = [x[neighbors] for neighbors in cyclic_neighbors]
        neighbor_num = torch.tensor([len(neighbors) for neighbors in cyclic_neighbors])
        padded_feats = pad_sequence(neighbor_feats, batch_first=False)
        packed_feats = pack_padded_sequence(padded_feats, neighbor_num, 
                                            batch_first=False, enforce_sorted=False)

        _, h_n = self.seq_model(packed_feats)
        seq_out = h_n[-1]

        return self.propagate(edge_index, x=x, seq=seq_out)

    def message(self, seq_j):
        return seq_j
    
    def aggregate(self, inputs, index, ptr = None, dim_size = None):
        return super().aggregate(inputs, index, ptr, dim_size)

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))
