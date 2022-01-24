from .self_attention import MultiHeadSelfAttentionBlock
from .feed_forward import PositionwiseFeedForwardBlock
from .positional_encoding import PositionalEncoding
import torch as th
import torch.nn as nn


class DecoderBlock(nn.Module):
    def __init__(self, vector_dim, embedding_dim, dropout=0.1, n_heads=8, feedforward_expansion=1) -> None:
        super().__init__()

        self.MaskedAttentionBlock = MultiHeadSelfAttentionBlock(vector_dim, embedding_dim, n_heads)
        self.MaskedAttentionNorm = nn.LayerNorm(vector_dim)

        self.AttentionBlock = MultiHeadSelfAttentionBlock(vector_dim, embedding_dim, n_heads)
        self.AttentionNorm = nn.LayerNorm(vector_dim)

        self.FeedForwardBlock = PositionwiseFeedForwardBlock(vector_dim, feedforward_expansion)
        self.FeedForwardNorm = nn.LayerNorm(vector_dim)

        self.Dropout = nn.Dropout(dropout)

    def forward(self, seqs, v_seqs, k_seqs, mask):
        """
        Args:
            seqs = sequences = [batch_size, sequence_len, vector_dim]
                input sequences

            v_seqs = [batch_size, sequence_len, vector_dim]
                output from encoder block

            k_seqs = [batch_size, sequence_len, vector_dim]
                output from encoder block
        Returns:
            out_seqs = out_sequences = [batch_size, sequence_len, vector_dim]
        """
        batch_size, seq_len, vector_dim = seqs.shape

        # ** attention using masked Attention **
        msk_att_seqs = self.MaskedAttentionBlock(seqs, seqs, seqs, mask)
        # apply drop defore add & norm
        msk_att_seqs = self.Dropout(msk_att_seqs)
        msk_att_seqs = self.MaskedAttentionNorm(msk_att_seqs + seqs)

        # ** cross attention with encoder output and use non-mask Attention **
        att_seqs = self.AttentionBlock(v_seqs, k_seqs, msk_att_seqs, None)
        # apply drop defore add & norm
        att_seqs = self.Dropout(att_seqs)
        att_seqs = self.AttentionNorm(att_seqs + msk_att_seqs)

        # ** feed foeward **
        forward_seqs = self.FeedForwardBlock(att_seqs)
        # apply drop defore add & norm
        forward_seqs = self.Dropout(forward_seqs)
        forward_seqs = self.FeedForwardNorm(forward_seqs + att_seqs)
        return forward_seqs


class Decoder(nn.Module):
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

        self.DecoderBlocks = nn.ModuleList(
            [DecoderBlock(vector_dim, embedding_dim, dropout, n_heads, feedforward_expansion) for _ in range(n_blocks)]
        )

    def forward(self, X, encoder_seqs, mask):
        """
        Args:
            x = sequences = [batch_size, sequence_len]
                input sequences to decode
            encoder_seqs = [batch_size, sequence_len, vector_dim]
                encoder output sequences
        Returns:
            out_seqs = out_sequences = [batch_size, sequence_len, vector_dim]
        """
        # seqs = [batch_size, sequence_len, vector_dim]
        seqs = self.WordEmbedding(X)
        seqs = self.PositionalEncoding(seqs)

        in_seqs = seqs
        for db in self.DecoderBlocks:
            in_seqs = db.forward(in_seqs, encoder_seqs, encoder_seqs, mask)

        out_seq = in_seqs
        return out_seq
