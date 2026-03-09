import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from Preprocessing.utils import data_loader_nap, preprocess_nap, check_gpu
from Preprocessing.loader import pad_sequences_pre
from Model.model import SiamSAEncoder, DownstreamClassifier # Import Classifier

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Configure experiment settings.')
parser.add_argument('--dataName', type=str, default='bpic13_o', help='Name of the dataset')
parser.add_argument('--strategy', type=str, default='combi', help='Strategy to use')
args = parser.parse_args()

dataName = args.dataName
BATCH_SIZE = 256
repetitions = 1
EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT = 128, 4, 256, 2, 0.2
HIDDEN_DIM, FEATURE_DIM = 256, 256

device = check_gpu()

data = data_loader_nap(dataName)
train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output, train_token_x, train_token_y = preprocess_nap(data)

for i in range(repetitions):
    train_x, val_x, train_y, val_y = train_test_split(
        train_token_x, train_token_y, test_size=0.15, random_state=42, shuffle=False
    )

    train_x_padded = pad_sequences_pre(train_x, maxlen=max_case_length) 
    val_x_padded = pad_sequences_pre(val_x, maxlen=max_case_length)

    train_ds = TensorDataset(torch.tensor(train_x_padded, dtype=torch.long), torch.tensor(train_y, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(val_x_padded, dtype=torch.long), torch.tensor(val_y, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
    pretrained_encoder = SiamSAEncoder(EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT, max_case_length, vocab_size, HIDDEN_DIM, FEATURE_DIM)
    pretrained_encoder.load_state_dict(torch.load(f"PreTrainedModels/{dataName}_pretrained.pth", map_location=device))
    
    model = DownstreamClassifier(pretrained_encoder, EMBED_DIM, len(y_word_dict), trainable=False).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_val_loss = float('inf')
    patience, patience_counter = 10, 0
    best_model_state = None

    for epoch in range(100):
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                val_loss += criterion(outputs, y_batch).item()
                
        val_loss /= len(val_loader)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

    model.load_state_dict(best_model_state)
    os.makedirs("NAPModels", exist_ok=True)
    torch.save(model.state_dict(), f"NAPModels/{dataName}_nap_{i}.pth")
    print("Model saved.")