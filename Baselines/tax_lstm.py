import torch
import torch.nn as nn

class TaxLSTM(nn.Module):
    def __init__(self, num_activities, max_seq_len, lstm_units=100, num_continuous_features=0):
        super(TaxLSTM, self).__init__()
        self.num_continuous = num_continuous_features
        
        # Activity embedding
        self.embedding = nn.Embedding(num_embeddings=num_activities + 1, embedding_dim=32)
        
        # Input dimension for LSTM
        input_dim = 32 + num_continuous_features
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_units, batch_first=True)
        
        self.batch_norm = nn.BatchNorm1d(lstm_units)
        self.fc = nn.Linear(lstm_units, num_activities)

    def forward(self, act_input, cont_input=None):
        x = self.embedding(act_input)
        
        # Concatenate continuous features (like time) if they exist
        if self.num_continuous > 0 and cont_input is not None:
            x = torch.cat((x, cont_input), dim=-1)
            
        out, (hn, cn) = self.lstm(x)
        
        # Extract the hidden state of the final time step
        out = out[:, -1, :] 
        out = self.batch_norm(out)
        logits = self.fc(out) 
        
        return logits