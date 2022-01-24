from torch._C import dtype
from models.components.encoder import Encoder
from models.components.decoder import Decoder
from models.transformer import Transformer

import torch as th
import torch.nn as nn


def test_usage():
    vector_dim = 512
    embedding_dim = 256
    n_heads = 2
    n_blocks = 2
    num_classes = 1000

    X = th.Tensor(
        [
            [1, 2, 3, 4, 4, 7, 5, 1, 5, 7, 4, 1, 5],
            [2, 3, 4, 5, 7, 4, 1, 5, 7, 8, 1, 42, 32],
            [3, 4, 5, 6, 4, 12, 4, 5, 5, 7, 4, 1, 5],
            [5, 6, 7, 8, 1, 42, 32, 10, 5, 7, 4, 1, 5],
            [6, 8, 10, 15, 5, 12, 11, 33, 5, 7, 4, 1, 5],
        ]
    ).int()

    T = Transformer(num_classes, vector_dim=vector_dim, embedding_dim=embedding_dim, n_heads=n_heads, n_blocks=n_blocks)
    y = T.forward(X, X, useMask=True)

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")


if __name__ == "__main__":
    test_usage()
