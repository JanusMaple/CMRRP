import os
import sys
import torch
from torch_geometric.data import HeteroData, InMemoryDataset

sys.path.append('..')
from EMRC import EMRC

class GENNData(HeteroData):
    def __inc__(self, key: str, value: any, store=None):
        if key.endswith('cyclic_neighbors'):
            if store is not None and hasattr(store, 'num_nodes'):
                return store.num_nodes
        return super().__inc__(key, value, store=store)

class GENNDataset(InMemoryDataset):
    def __init__(self, emrc_pairs: list[tuple[EMRC, EMRC, torch.tensor]], 
                 root = None, force_reload = False):
        self.emrc_pairs = emrc_pairs
        if root is None:
            root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'data'
            )
        super().__init__(root, force_reload=force_reload, 
                         transform=None, pre_transform=None, pre_filter=None)
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
            data = GENNData()
            data['emrc_1'].x = x_1
            data['emrc_1', 'emrc_1_edges', 'emrc_1'].edge_index = edge_index_1
            data['emrc_1'].cyclic_neighbors = cyclic_neighbors_1
            data['emrc_1'].neighbor_num = neighbor_num_1
            data['emrc_1'].num_nodes = x_1.size()[0]
            data['emrc_2'].x = x_2
            data['emrc_2', 'emrc_2_edges', 'emrc_2'].edge_index = edge_index_2
            data['emrc_2'].cyclic_neighbors = cyclic_neighbors_2
            data['emrc_2'].neighbor_num = neighbor_num_2
            data['emrc_2'].num_nodes = x_2.size()[0]
            data.distance = distance
            data_list.append(data)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
