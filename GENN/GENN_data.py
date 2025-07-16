import os
import sys
import torch
from torch_geometric.data import Data, InMemoryDataset, DataLoader

sys.path.append('..')
from EMRC import EMRC

class GENNData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'cyclic_neighbors':
            return self.num_nodes
        return super().__inc__(key, value, *args, **kwargs)

class GENNDataset(InMemoryDataset):
    def __init__(self, emrcs: list[EMRC], root = None):
        if root is None:
            root = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                'data'
            )
        super().__init__(root, transform=None, pre_transform=None, pre_filter=None)
        self.emrcs = emrcs
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
        for emrc in self.emrcs:
            x = torch.ones(2 * emrc.m - 2 * emrc.w - emrc.v)
            


        # data_list = []
        # for g in self.graphs:
        #     x = torch.tensor(g['x'], dtype=torch.float)
        #     edge_index = torch.tensor(g['edge_index'], dtype=torch.long)
        #     cyclic_neighbors = [torch.tensor(nb, dtype=torch.long) for nb in g['cyclic_neighbors']]
        #     # Store as attribute on Data object
        #     data = GENNData(
        #         x=x, 
        #         edge_index=edge_index, 
        #         cyclic_neighbors=cyclic_neighbors
        #         )
        #     if 'y' in g:
        #         data.y = torch.tensor(g['y'])
        #     data_list.append(data)

        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])
