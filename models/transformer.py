from .components import Encoder, Decoder
import torch as th
import torch.nn as nn
import math as m


class Transformer(nn.Module):
    def __init__(
        self,
        output_dim,
        pad_index=0,
        n_blocks=6,
        vector_dim=512,
        embedding_dim=512,
        max_vocab_num=4096,
        dropout=0.1,
        n_heads=8,
        feedforward_expansion=4,
    ) -> None:
        super().__init__()
        self.pad_index = pad_index

        self.Encoder = Encoder(n_blocks, vector_dim, embedding_dim, max_vocab_num, dropout, n_heads, feedforward_expansion)
        self.Decoder = Decoder(n_blocks, vector_dim, embedding_dim, max_vocab_num, dropout, n_heads, feedforward_expansion)

        self.OutFC = nn.Sequential(
            nn.Linear(vector_dim, output_dim),
            nn.Softmax(-1),
        )

    def generateMasks(self, encoder_X, decoder_X, useMask):
        """
        Args:
            encoder_X = [batch_size, sequence_len]
            decoder_X = [batch_size, sequence_len]

        Returns
            encoder_mask = [sequence_len, sequence_len]
            decoder_mask = [sequence_len, sequence_len]
        """
        if useMask:
            _, es = encoder_X.shape
            _, ds = decoder_X.shape

            encoder_mask = th.ones(es, es)
            decoder_mask = th.tril(th.ones(ds, ds))
            return encoder_mask, decoder_mask
        return None, None

    def forward(self, scr_X, tar_X, useMask=False):
        encoder_mask, decoder_mask = self.generateMasks(scr_X, tar_X, useMask)

        encoder_out_seqs = self.Encoder.forward(scr_X, encoder_mask)
        decoder_out_seqs = self.Decoder.forward(tar_X, encoder_out_seqs, decoder_mask)

        mean_seqs = decoder_out_seqs.mean(axis=1)
        output = self.OutFC(mean_seqs)
        return output
