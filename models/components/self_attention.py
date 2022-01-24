import torch as th
import math
import torch.nn as nn


class ScaledDotProductAttentionBlock(nn.Module):
    def __init__(self, vector_dim) -> None:
        super().__init__()
        self.vector_dim = vector_dim  # how many features in a vector
        self.scale = 1 / math.sqrt(self.vector_dim)

        self.Softmax = nn.Softmax(-1)

    def forward(self, V: th.Tensor, K: th.Tensor, Q: th.Tensor, mask: th.Tensor):
        """
        Args:
            v_in = [batch_size, sequences_len, vector_dim]
            k_in = [batch_size, sequences_len, vector_dim]
            q_in = [batch_size, sequences_len, vector_dim]
        Returns:
            v_out = [batch_size, sequences_len, vector_dim]
        """

        # A shape [batch_size, sequences_len, sequences_len]
        A = Q @ K.transpose(1, 2) * self.scale  # scaled product attention

        if mask is not None:
            A = A.masked_fill(mask == 0, 1e-20)  # mask attention

        A = self.Softmax(A)

        one_v_out = A @ V
        return one_v_out


class MultiHeadSelfAttentionBlock(nn.Module):
    def __init__(self, q_vector_dim, kv_vector_dim, embedding_dim, n_heads) -> None:
        super().__init__()
        self.q_vector_dim = q_vector_dim  # how many features in a vector
        self.kv_vector_dim = kv_vector_dim
        self.n_heads = n_heads
        self.head_dim = embedding_dim // n_heads

        assert (
            self.head_dim * n_heads == embedding_dim
        ), f"embedding_dim: {embedding_dim} needs to be divided by n_heads: {n_heads}"

        self.SelfAttentionBlocks = nn.ModuleList(
            [ScaledDotProductAttentionBlock(self.head_dim) for _ in range(self.n_heads)]
        )

        self.Value = nn.Linear(kv_vector_dim, embedding_dim, bias=False)
        self.Key = nn.Linear(kv_vector_dim, embedding_dim, bias=False)
        self.Query = nn.Linear(q_vector_dim, embedding_dim, bias=False)

        self.OutFC = nn.Linear(embedding_dim, self.q_vector_dim, bias=False)

    def forward(self, v_in, k_in, q_in, mask):
        """
        Args:
            v_in = [batch_size, kv_sequences_len, kv_vector_dim]
            k_in = [batch_size, kv_sequences_len, kv_vector_dim]
            q_in = [batch_size, q_sequences_len, q_vector_dim]
        """
        b, kv_s, _ = v_in.shape
        b, q_s, _ = q_in.shape

        # project K Q V into embedding_dim first
        V = self.Value(v_in)  # [batch_size, sequences_len, embedding_dim]
        K = self.Key(k_in)  # [batch_size, sequences_len, embedding_dim]
        Q = self.Query(q_in)  # [batch_size, sequences_len, embedding_dim]

        # cut them into heads
        # reshape and permute input shapes to [heads, batch, sequence_len, head_dim]
        v_heads = V.reshape(b, kv_s, self.n_heads, self.head_dim).permute(2, 0, 1, 3)
        k_heads = K.reshape(b, kv_s, self.n_heads, self.head_dim).permute(2, 0, 1, 3)
        q_heads = Q.reshape(b, q_s, self.n_heads, self.head_dim).permute(2, 0, 1, 3)

        # # multihead attention, attention on each head
        if mask is not None:
            assert (q_s, kv_s) == mask.shape

        v_out = []
        for h in range(self.n_heads):

            # pass in [batch, sequence_len, head_dim]
            one_v_out = self.SelfAttentionBlocks[h].forward(v_heads[h], k_heads[h], q_heads[h], mask)
            v_out.append(one_v_out)

        # combine each head together
        # v_out = [n_heads, batch_size, sequence_len, head_dim] => [batch, sequence_len, vector_dim]
        v_out = th.cat(v_out, axis=2)

        # project V back to vector dim
        # v_out = [batch, sequence_len, vector_dim]
        v_out = self.OutFC(v_out)
        return v_out
