"""
To prepare this script we used the following source codes:
    https://github.com/rampasek/GraphGPS
We adjusted the source codes to efficiently integrate them into our framework.
"""

from torch_geometric.graphgym.register import register_config


@register_config('custom_gnn')
def custom_gnn_cfg(cfg):
    """Extending config group of GraphGym's built-in GNN for purposes of our
    CustomGNN network model.
    """

    # Use residual connections between the GNN layers.
    cfg.gnn.residual = False
