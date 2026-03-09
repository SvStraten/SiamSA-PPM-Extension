import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

from Preprocessing.utils import data_loader_nap, preprocess_nap, check_gpu
from Preprocessing.loader import pad_sequences_pre
from Model.model import SiamSAEncoder, DownstreamClassifier
from Augmentation.kappel_augmentation import apply_kappel_eda

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Configure Kappel baseline experiment settings.')
parser.add_argument('--dataName', type=str, default='bpic13_o', help='Name of the dataset')
parser.add_argument('--factor', type=float, default=1.2, help='Augmentation factor for Kappel EDA')
args = parser.parse_args()

dataName = args.dataName
AUG_FACTOR = args.factor
BATCH_SIZE = 256
repetitions = 1

# Architecture Hyperparameters
EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT = 128, 4, 256, 2, 0.2
HIDDEN_DIM, FEATURE_DIM = 256, 256

device = check_gpu()

# 1. Load Data
data = data_loader_nap(dataName)
train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output, train_token_x, train_token_y = preprocess_nap(data)

for i in range(repetitions):
    print(f"\n=== Starting Repetition {i+1}/{repetitions} for {dataName} (Kappel EDA Factor: {AUG_FACTOR}x) ===")
    
    # 2. Split Data
    train_x, val_x, train_y, val_y = train_test_split(
        train_token_x, train_token_y, test_size=0.15, random_state=42, shuffle=False
    )

    # 3. Apply Kappel & Jablonski EDA Strategy
    train_x_expanded, train_y_expanded = apply_kappel_eda(
        train_x, 
        train_y, 
        x_word_dict, 
        augmentation_factor=AUG_FACTOR
    )

    # 4. Pad Sequences (Pre-padding)
    train_x_padded = pad_sequences_pre(train_x_expanded, maxlen=max_case_length) 
    val_x_padded = pad_sequences_pre(val_x, maxlen=max_case_length)

    # 5. Create PyTorch DataLoaders
    train_ds = TensorDataset(torch.tensor(train_x_padded, dtype=torch.long), torch.tensor(train_y_expanded, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(val_x_padded, dtype=torch.long), torch.tensor(val_y, dtype=torch.long))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
        
    # 6. Initialize FRESH Backbone (No BYOL Pretraining for this Baseline)
    encoder = SiamSAEncoder(EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT, max_case_length, vocab_size, HIDDEN_DIM, FEATURE_DIM)
    
    # Pass to DownstreamClassifier with trainable=True so the whole network learns from scratch
    model = DownstreamClassifier(encoder, EMBED_DIM, len(y_word_dict), trainable=True).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    # 7. Training Loop with Early Stopping
    best_val_loss = float('inf')
    patience, patience_counter = 10, 0
    best_model_state = None

    for epoch in range(100):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Validation Step
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                outputs = model(x_batch)
                val_loss += criterion(outputs, y_batch).item()
                
        val_loss /= len(val_loader)
        
        # Early Stopping Check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
                
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/100] | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

    # 8. Restore best weights and save the Kappel Baseline model
    model.load_state_dict(best_model_state)
    os.makedirs("NAPModels", exist_ok=True)
    
    save_path = f"NAPModels/{dataName}_kappel_nap_{i}.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully at {save_path}.")