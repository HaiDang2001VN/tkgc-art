import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalTransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, dim_feedforward, dropout=0.1, max_seq_length=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        # Token embedding and positional encoding
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.positional_embedding = nn.Embedding(max_seq_length, embed_dim)

        # Create encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=False
        )
        
        # Create transformer encoder with specified number of layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Final projection to vocabulary size
        self.fc_out = nn.Linear(embed_dim, vocab_size)
        
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialize parameters following Transformer paper recommendations
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """
        x: [seq_len, batch_size] or [seq_len, batch_size, embed_dim]
        """
        seq_len, batch_size = x.shape[0], x.shape[1]
        
        # Handle different input types (token indices or embeddings)
        if x.dim() == 2:  # [seq_len, batch_size]
            # Create position indices
            positions = torch.arange(0, seq_len, device=x.device).unsqueeze(1).expand(seq_len, batch_size)
            
            # Embed tokens and positions, then add them
            token_emb = self.token_embedding(x)
            pos_emb = self.positional_embedding(positions)
            x = token_emb + pos_emb
        # else it's already [seq_len, batch_size, embed_dim]
        
        x = self.dropout(x)
        
        # Apply transformer encoder with causal attention
        # is_causal=True applies causal masking automatically
        output = self.transformer_encoder(x, is_causal=True)
        
        # Project to vocabulary size
        logits = self.fc_out(output)  # [seq_len, batch_size, vocab_size]
        return logits

    def generate(self, prompt, max_new_tokens, temperature=1.0):
        """
        Autoregressive generation with the causal transformer model
        prompt: [prompt_len, batch_size] or [prompt_len, batch_size, embed_dim]
        """
        self.eval()
        
        # If prompt is token IDs, keep as is
        if prompt.dim() == 2:
            seq = prompt.clone()
        # If prompt is embeddings, we need to work with the embedded space
        else:
            # This is a placeholder - in a real implementation you'd need to
            # handle working with embeddings directly or converting them back to tokens
            raise NotImplementedError("Generation from embeddings not implemented")
        
        for _ in range(max_new_tokens):
            # Forward pass with sequence so far
            logits = self(seq)
            
            # Get logits for the last token and apply temperature
            next_token_logits = logits[-1, :, :] / temperature  # [batch_size, vocab_size]
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).transpose(0, 1)  # [1, batch_size]
            
            # Append to the sequence
            seq = torch.cat([seq, next_token], dim=0)
            
            # Break if we exceed maximum sequence length
            if seq.size(0) >= self.max_seq_length:
                break
                
        return seq
