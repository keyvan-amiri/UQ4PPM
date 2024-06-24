"""
To prepare this script we used the following source codes:
    https://github.com/rampasek/GraphGPS
We adjusted the source codes to efficiently integrate them into our framework.
"""

import torch
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import (register_node_encoder,
                                               register_edge_encoder)

"""
Generic Node and Edge encoders for datasets with node/edge features that
consist of only one type dictionary thus require a single nn.Embedding layer.

The number of possible Node and Edge types must be set by cfg options:
1) cfg.dataset.node_encoder_num_types
2) cfg.dataset.edge_encoder_num_types

In case of a more complex feature set, use a data-specific encoder.

"""


@register_node_encoder('TypeDictNode')
class TypeDictNodeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = cfg.dataset.node_encoder_num_types
        if num_types < 1:
            raise ValueError(f"Invalid 'node_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        # Encode just the first dimension if more exist
        batch.x = self.encoder(batch.x[:, 0])

        return batch


@register_edge_encoder('TypeDictEdge')
class TypeDictEdgeEncoder(torch.nn.Module):
    def __init__(self, emb_dim):
        super().__init__()

        num_types = cfg.dataset.edge_encoder_num_types
        if num_types < 1:
            raise ValueError(f"Invalid 'edge_encoder_num_types': {num_types}")

        self.encoder = torch.nn.Embedding(num_embeddings=num_types,
                                          embedding_dim=emb_dim)
        # torch.nn.init.xavier_uniform_(self.encoder.weight.data)

    def forward(self, batch):
        batch.edge_attr = self.encoder(batch.edge_attr)
        return batch
