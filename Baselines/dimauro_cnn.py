import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock1D, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding='same')
        self.conv5 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding='same')
        
        self.pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pool_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same')

    def forward(self, x):
        c1 = F.relu(self.conv1(x))
        c3 = F.relu(self.conv3(x))
        c5 = F.relu(self.conv5(x))
        p = self.pool(x)
        pc = F.relu(self.pool_conv(p))
        return torch.cat([c1, c3, c5, pc], dim=1)

class DiMauroIDCNN(nn.Module):
    def __init__(self, num_activities, max_seq_len, filters=32):
        super(DiMauroIDCNN, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=num_activities + 1, embedding_dim=32)
        
        # The embedding outputs 32 channels. 
        self.inc1 = InceptionBlock1D(in_channels=32, out_channels=filters)
        self.inc2 = InceptionBlock1D(in_channels=filters * 4, out_channels=filters)
        self.inc3 = InceptionBlock1D(in_channels=filters * 4, out_channels=filters)
        
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(filters * 4, num_activities)

    def forward(self, x):
        x = self.embedding(x)  # Shape: (Batch, Seq, 32)
        x = x.transpose(1, 2)  # Reshape for PyTorch Conv1D: (Batch, 32, Seq)
        
        x = self.inc1(x)
        x = self.inc2(x)
        x = self.inc3(x)
        
        # Global Max Pooling 1D
        x = torch.max(x, dim=2)[0] 
        
        x = self.dropout(x)
        logits = self.fc(x)
        return logits