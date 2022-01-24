import torch.nn as nn
import math as m
import torch as th


class PositionalEncoding(nn.Module):
    def __init__(self, dropout) -> None:
        super().__init__()
        self.Dropout = nn.Dropout(dropout)

    def encode_even_func(self, pos, dim, vector_dim):
        return m.sin(pos / 10000 ** (2 * dim / vector_dim))

    def encode_odd_func(self, pos, dim, vector_dim):
        return m.cos(pos / 10000 ** (2 * dim / vector_dim))

    def encode(self, seqs):
        _, seq_len, vector_dim = seqs.shape
        # we can store these values before training

        position_code = []
        for pos in range(seq_len):
            vector_code = []
            for dim in range(vector_dim):
                if dim % 2:
                    vector_code.append(self.encode_odd_func(pos, dim, vector_dim))
                else:
                    vector_code.append(self.encode_even_func(pos, dim, vector_dim))
            position_code.append(vector_code)
        position_code = th.Tensor(position_code).expand(seqs.shape)

        # encode sequences
        seqs += position_code
        seqs = self.Dropout(seqs)
        return seqs

    def forward(self, seqs):
        """
        Args:
            seqs = [batch_size, sequence_len, vector_dim]

        Returns:
            seqs = [batch_size, sequence_len, vector_dim]
                with postional encoding
        """
        assert len(seqs.shape) == 3, "You need to embed sequence to shape [batch_size, sequence_len, vector_dim]"

        seqs = self.encode(seqs)
        return seqs
