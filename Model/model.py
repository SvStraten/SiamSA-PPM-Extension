import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        # padding_idx=0 ensures that the embedding vector for padding is all zeros
        self.token_emb = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(num_embeddings=maxlen, embedding_dim=embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        # Generate position indices [0, 1, ..., seq_len-1] and expand to batch size
        positions = torch.arange(0, seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        positions = self.pos_emb(positions)
        
        x = self.token_emb(x)
        return x + positions


class SiamSAEncoder(nn.Module):
    """
    Combines the Transformer Encoder backbone and the Projection MLP.
    """
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout, maxlen, vocab_size, hidden_dim, feature_dim):
        super(SiamSAEncoder, self).__init__()
        self.embedding = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
        
        # PyTorch's native Transformer block
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Projection MLP (z_theta / z_xi)
        self.projector = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim, bias=False),
            nn.BatchNorm1d(feature_dim)
        )

    def forward(self, x):
        # Create a boolean mask for padding tokens.
        # PyTorch expects True for positions that should be IGNORED (padded values).
        # Assuming 0 is the padding token.
        pad_mask = (x == 0)
        
        # Embed and pass through transformer
        emb = self.embedding(x)
        encoded = self.transformer_blocks(emb, src_key_padding_mask=pad_mask)
        
        # Global Average Pooling 1D (masking out the padding tokens)
        # Invert the pad_mask (1.0 for real tokens, 0.0 for padding)
        valid_mask = (~pad_mask).unsqueeze(-1).float() 
        
        # Sum only the valid token embeddings and divide by the actual sequence length per batch
        x_sum = torch.sum(encoded * valid_mask, dim=1)
        x_lengths = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
        x_pool = x_sum / x_lengths
        
        # Pass through the projection MLP
        projected = self.projector(x_pool)
        return projected


class SiamSAPredictor(nn.Module):
    """
    The prediction MLP applied to the online network to prevent representation collapse.
    """
    def __init__(self, feature_dim):
        super(SiamSAPredictor, self).__init__()
        self.predictor = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 4, bias=False),
            nn.BatchNorm1d(feature_dim // 4),
            nn.ReLU(),
            nn.Linear(feature_dim // 4, feature_dim, bias=True)
        )

    def forward(self, x):
        return self.predictor(x)
    

class DownstreamClassifier(nn.Module):
    def __init__(self, encoder_backbone, embed_dim, num_classes, trainable=False):
        super(DownstreamClassifier, self).__init__()
        self.encoder = encoder_backbone
        # Freeze or unfreeze the backbone
        for param in self.encoder.parameters():
            param.requires_grad = trainable
            
        self.fc = nn.Linear(embed_dim, num_classes)
        
    def forward(self, x):
        pad_mask = (x == 0)
        emb = self.encoder.embedding(x)
        encoded = self.encoder.transformer_blocks(emb, src_key_padding_mask=pad_mask)
        
        valid_mask = (~pad_mask).unsqueeze(-1).float() 
        x_sum = torch.sum(encoded * valid_mask, dim=1)
        x_lengths = torch.clamp(valid_mask.sum(dim=1), min=1e-9)
        x_pool = x_sum / x_lengths
        
        return self.fc(x_pool)