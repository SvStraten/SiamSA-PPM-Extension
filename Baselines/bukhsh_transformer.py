import torch
import torch.nn as nn

class TokenAndPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

    def forward(self, x):
        seq_len = x.size(1)
        # Generate position indices [0, 1, ..., seq_len-1]
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        return self.token_emb(x) + self.pos_emb(positions)

class BukhshTransformer(nn.Module):
    def __init__(self, num_activities, max_seq_len, embed_dim=32, num_heads=2, ff_dim=32):
        super(BukhshTransformer, self).__init__()
        self.embedding = TokenAndPositionEmbedding(max_seq_len, num_activities + 1, embed_dim)
        
        # Built-in PyTorch Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=ff_dim, 
            dropout=0.1, 
            batch_first=True
        )
        self.transformer_block = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        self.dropout1 = nn.Dropout(0.2)
        self.fc1 = nn.Linear(embed_dim, 64)
        self.dropout2 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_activities)

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer_block(x)
        
        # Global Average Pooling 1D (Mean across the sequence dimension)
        x = torch.mean(x, dim=1)
        
        x = self.dropout1(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout2(x)
        logits = self.fc2(x)
        return logits