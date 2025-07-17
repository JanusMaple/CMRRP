import os
import sys
import torch
from torch_geometric.data import Data, InMemoryDataset, Batch

sys.path.append('..')
from EMRC import EMRC

class GENNData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cyclic_neighbors':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

class GENNDataset(InMemoryDataset):
    def __init__(self, emrc_pairs: list[tuple[EMRC, EMRC, torch.tensor]], root = None):
        if root is None:
            root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'data'
            )
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)
        self.emrc_pairs = emrc_pairs
        self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []                                   # No raw files are needed

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        pass                                        # TODO: Upload dataset to internet

    def process(self):
        data_list = []
        for emrc_pair in self.emrc_pairs:
            emrc_1, emrc_2, distance = emrc_pair
            x_1, edge_index_1, cyclic_neighbors_1, neighbor_num_1 = \
                emrc_1.get_representation(True)
            x_2, edge_index_2, cyclic_neighbors_2, neighbor_num_2 = \
                emrc_2.get_representation(True)
            data = GENNData(
                x_1 = x_1, edge_index_1 = edge_index_1, 
                cyclic_neighbors_1 = cyclic_neighbors_1, neighbor_num_1 = neighbor_num_1,
                x_2 = x_2, edge_index_2 = edge_index_2, 
                cyclic_neighbors_2 = cyclic_neighbors_2, neighbor_num_2 = neighbor_num_2,
                distance = distance)                # distance: torch.tensor([float])
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class GENNDataBatch(Batch):
    def __init__(self, batch):
        pass
