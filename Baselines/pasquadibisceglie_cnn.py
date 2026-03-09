import torch
import torch.nn as nn
import torch.nn.functional as F

class Pasquadibisceglie2DCNN(nn.Module):
    def __init__(self, num_activities):
        super(Pasquadibisceglie2DCNN, self).__init__()
        # In PyTorch, images are (Channels, Height, Width). We have 1 channel.
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding='same')
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same')
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.flatten = nn.Flatten()
        
        # LazyLinear automatically infers the input feature size upon the first forward pass
        self.fc1 = nn.LazyLinear(out_features=128)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, num_activities)

    def forward(self, x):
        # x expected shape: (Batch, 1, Seq_Len, Num_Features)
        x = F.relu(self.conv1(x))
        x = F.pad(x, (0, 1, 0, 1)) # Simple pad to accommodate pooling of odd dimensions
        x = self.pool1(x)
        
        x = F.relu(self.conv2(x))
        x = F.pad(x, (0, 1, 0, 1))
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits