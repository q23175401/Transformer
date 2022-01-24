from .self_attention import MultiHeadSelfAttentionBlock
from .feed_forward import PositionwiseFeedForwardBlock
from .positional_encoding import PositionalEncoding
import torch as th
import torch.nn as nn


class EncoderBlock(nn.Module):
    def __init__(self, vector_dim, embedding_dim, dropout=0.1, n_heads=8, feedforward_expansion=1) -> None:
        super().__init__()

        self.AttentionBlock = MultiHeadSelfAttentionBlock(vector_dim, embedding_dim, n_heads)
        self.AttentionNorm = nn.LayerNorm(vector_dim)

        self.FeedForwardBlock = PositionwiseFeedForwardBlock(vector_dim, feedforward_expansion)
        self.FeedForwardNorm = nn.LayerNorm(vector_dim)

        self.Dropout = nn.Dropout(dropout)

    def forward(self, seqs, mask):
        """
        Args:
            seqs = sequences = [batch_size, sequence_len, vector_dim]

        Returns:
            out_seqs = out_sequences = [batch_size, sequence_len, vector_dim]
        """
        batch_size, seq_len, vector_dim = seqs.shape

        # Attention
        att_seqs = self.AttentionBlock(seqs, seqs, seqs, mask)
        # apply dropout before add & norm
        att_seqs = self.Dropout(att_seqs)
        att_seqs = self.AttentionNorm((att_seqs + seqs).reshape(batch_size * seq_len, vector_dim))
        att_seqs = att_seqs.reshape(batch_size, seq_len, vector_dim)

        # Feed Forward
        forward_seqs = self.FeedForwardBlock(att_seqs)
        # apply dropout before add & norm
        forward_seqs = self.Dropout(forward_seqs)
        forward_seqs = self.FeedForwardNorm((forward_seqs + att_seqs).reshape(batch_size * seq_len, vector_dim))

        out_seqs = forward_seqs.reshape(batch_size, seq_len, vector_dim)
        return out_seqs


class Encoder(nn.Module):
    def __init__(
        self,
        n_blocks=6,
        vector_dim=512,
        embedding_dim=512,
        max_vocab_num=4096,
        dropout=0.1,
        n_heads=8,
        feedforward_expansion=4,
    ) -> None:
        super().__init__()

        self.WordEmbedding = nn.Embedding(max_vocab_num, vector_dim)
        self.PositionalEncoding = PositionalEncoding(dropout)

        self.EncoderBlocks = nn.ModuleList(
            [EncoderBlock(vector_dim, embedding_dim, dropout, n_heads, feedforward_expansion) for _ in range(n_blocks)]
        )

    def forward(self, X, mask):
        """
        Args:
            X = [batch_size, sequence_len]

        Returns:
            out_seqs = out_sequences = [batch_size, sequence_len, vector_dim]
        """
        # seqs = [batch_size, sequence_len, vector_dim]
        seqs = self.WordEmbedding(X)
        seqs = self.PositionalEncoding(seqs)

        for eb in self.EncoderBlocks:
            seqs = eb.forward(seqs, mask)

        out_seq = seqs
        return out_seq
