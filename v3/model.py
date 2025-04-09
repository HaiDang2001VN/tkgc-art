import torch
import torch.nn as nn
import torch.nn.functional as F


def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout=0.1, max_seq_length=512):
        super(TransformerModel, self).__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_length, embed_dim)

        # Transformer (encoder-decoder)
        self.transformer = nn.Transformer(d_model=embed_dim,
                                          nhead=nhead,
                                          num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers,
                                          dim_feedforward=dim_feedforward,
                                          dropout=dropout)
        # Final linear projection to vocabulary size
        self.fc_out = nn.Linear(embed_dim, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize parameters following Transformer paper recommendations.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        src: [src_seq_len, batch_size] integer tokens
        tgt: [tgt_seq_len, batch_size] integer tokens
        """
        src_seq_len, batch_size = src.shape
        tgt_seq_len, _ = tgt.shape

        # Create position indices
        src_positions = torch.arange(0, src_seq_len, device=src.device).unsqueeze(
            1).expand(src_seq_len, batch_size)
        tgt_positions = torch.arange(0, tgt_seq_len, device=tgt.device).unsqueeze(
            1).expand(tgt_seq_len, batch_size)

        # Embed tokens and positions, then add them
        src_emb = self.token_embedding(
            src) + self.positional_embedding(src_positions)
        tgt_emb = self.token_embedding(
            tgt) + self.positional_embedding(tgt_positions)

        src_emb = self.dropout(src_emb)
        tgt_emb = self.dropout(tgt_emb)

        # If no masks provided, create default target mask (for causal masking)
        if tgt_mask is None:
            tgt_mask = generate_square_subsequent_mask(
                tgt_seq_len).to(src.device)

        # Transformer expects shape [seq_len, batch_size, embed_dim]
        transformer_output = self.transformer(
            src_emb, tgt_emb, src_mask=src_mask, tgt_mask=tgt_mask)

        # Project to vocabulary size
        # shape: [tgt_seq_len, batch_size, vocab_size]
        logits = self.fc_out(transformer_output)
        return logits

    def generate(self, src, max_new_tokens, start_symbol):
        """
        Autoregressive generation: given a source, generate tokens step-by-step.
        src: [src_seq_len, batch_size]
        Returns generated token ids: [generated_seq_len, batch_size]
        """
        self.eval()
        src_seq_len, batch_size = src.shape
        src_mask = None  # assume full attention on encoder side
        # Encode source once
        src_positions = torch.arange(0, src_seq_len, device=src.device).unsqueeze(
            1).expand(src_seq_len, batch_size)
        src_emb = self.token_embedding(
            src) + self.positional_embedding(src_positions)
        memory = self.transformer.encoder(src_emb, mask=src_mask)

        # Start generation with the start_symbol token
        generated = torch.full((1, batch_size), start_symbol,
                               dtype=torch.long, device=src.device)

        for _ in range(max_new_tokens):
            tgt_seq_len = generated.shape[0]
            tgt_mask = generate_square_subsequent_mask(
                tgt_seq_len).to(src.device)
            tgt_positions = torch.arange(0, tgt_seq_len, device=src.device).unsqueeze(
                1).expand(tgt_seq_len, batch_size)
            tgt_emb = self.token_embedding(
                generated) + self.positional_embedding(tgt_positions)
            tgt_emb = self.dropout(tgt_emb)
            decoder_output = self.transformer.decoder(
                tgt_emb, memory, tgt_mask=tgt_mask)
            # [tgt_seq_len, batch_size, vocab_size]
            logits = self.fc_out(decoder_output)
            # Get last time step logits and apply softmax
            last_logits = logits[-1]  # [batch_size, vocab_size]
            probs = F.softmax(last_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).transpose(
                0, 1)  # shape: [1, batch_size]
            generated = torch.cat([generated, next_tokens], dim=0)
        return generated
