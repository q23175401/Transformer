from typing import Sequence
import torch as th
import torch.nn as nn


class PositionwiseFeedForwardBlock(nn.Module):
    def __init__(self, vector_dim, forward_expansion=4) -> None:
        super().__init__()

        self.feed = nn.Sequential(
            nn.Linear(vector_dim, forward_expansion * vector_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * vector_dim, vector_dim),
        )

    def forward(self, sequences):
        """
        Args:
            sequences = [batchsize, sequence_len, vector_dim]
        """
        s_out = self.feed(sequences)

        return s_out
