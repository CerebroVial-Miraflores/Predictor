"""
Time-then-Space Spatiotemporal Graph Neural Network model.
"""

import torch
import torch.nn as nn
from typing import Dict, Any

from tsl.nn.blocks import RNN, MLPDecoder
from tsl.nn.layers import NodeEmbedding, DiffConv


class TimeThenSpaceModel(nn.Module):
    """
    Spatiotemporal Graph Neural Network with Time-then-Space architecture.
    """
    
    def __init__(
        self,
        input_size: int,
        n_nodes: int,
        horizon: int,
        emb_size: int = 16,
        hidden_size: int = 32,
        rnn_layers: int = 1,
        gnn_kernel: int = 2,
        rnn_cell: str = 'gru',
        dropout: float = 0.0
    ):
        super(TimeThenSpaceModel, self).__init__()
        
        self.input_size = input_size
        self.n_nodes = n_nodes
        self.horizon = horizon
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        
        # Node embeddings
        self.node_embeddings = NodeEmbedding(n_nodes, emb_size)
        
        # Encoder
        self.encoder = nn.Linear(input_size + emb_size, hidden_size)
        
        # Temporal processor: RNN
        self.time_nn = RNN(
            input_size=hidden_size,
            hidden_size=hidden_size,
            n_layers=rnn_layers,
            cell=rnn_cell,
            return_only_last_state=True,
            dropout=dropout if rnn_layers > 1 else 0.0
        )
        
        # Spatial processor: Diffusion Convolution
        self.space_nn = DiffConv(
            in_channels=hidden_size,
            out_channels=hidden_size,
            k=gnn_kernel,
            root_weight=True
        )
        
        # Decoder
        self.decoder = MLPDecoder(
            input_size=hidden_size + emb_size,
            hidden_size=2 * hidden_size,
            output_size=input_size,
            horizon=horizon,
            n_layers=1,
            dropout=dropout
        )
    
    def forward(self, x, edge_index, edge_weight):
        """Forward pass of the model."""
        b, t, n, f = x.size()
        
        # Concatenate node embeddings to input
        emb = self.node_embeddings(expand=(b, t, -1, -1))
        x_emb = torch.cat([x, emb], dim=-1)
        
        # Encode
        x_enc = self.encoder(x_emb)
        
        # Temporal processing
        h = self.time_nn(x_enc)
        
        # Spatial processing
        z = self.space_nn(h, edge_index, edge_weight)
        
        # Decode
        emb = self.node_embeddings(expand=(b, -1, -1))
        z_emb = torch.cat([z, emb], dim=-1)
        x_out = self.decoder(z_emb)
        
        return x_out