import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import random

# UPDATED IMPORTS: Pointing to the new consolidated file
from Augmentation.augmentation_module import (
    get_patterns, map_patterns_to_tokens,
    get_xor_candidates, map_xor_candidates_to_tokens,
    RandomDeletion, RandomInsertion, RandomReplacement,
    StatisticalInsertion, StatisticalDeletion, StatisticalReplacement
)
from Preprocessing.utils import data_loader_nap, preprocess_nap, check_gpu, check_applicability, generate_augmented_views
from Model.model import SiamSAEncoder, SiamSAPredictor

parser = argparse.ArgumentParser(description='Configure experiment settings.')
parser.add_argument('--dataName', type=str, default='sepsis', help='Name of the dataset')
parser.add_argument('--strategy', type=str, default='combi', help='Strategy to use')
args = parser.parse_args()

dataName = args.dataName
STRATEGY = args.strategy

# Hyperparameters
ALPHA, BETA, GAMMA, DELTA = 0.0001, 0.0001, 0.0001, 0.0001
PATH_LENGTH = 4
BATCH_SIZE = 256
EPOCHS = 100
WARMUP_EPOCHS = 10
BASE_LR = 0.05 
WEIGHT_DECAY = 1e-5
TAU_BASE = 0.996

EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT = 128, 4, 256, 2, 0.2
HIDDEN_DIM, FEATURE_DIM = 256, 256

device = check_gpu()

data = data_loader_nap(dataName)
train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output, train_token_x, train_token_y = preprocess_nap(data)
available_tokens = list(x_word_dict.values())

patterns_df = get_patterns(f'datasets/{dataName}/processed/next_activity_train.csv', transition_threshold=BETA, path_threshold=GAMMA, max_path_length=PATH_LENGTH, activity_threshold=ALPHA)
patterns_token_df = map_patterns_to_tokens(patterns_df, x_word_dict)

xor_df = get_xor_candidates(f'datasets/{dataName}/processed/next_activity_train.csv', support_threshold=DELTA, max_path_length=PATH_LENGTH, activity_threshold=ALPHA)
xor_token_df = map_xor_candidates_to_tokens(xor_df, x_word_dict)

if STRATEGY == 'random':
    main_augmentors = []
elif STRATEGY == "combi":
    main_augmentors = [
        StatisticalInsertion(patterns_token_df),
        StatisticalDeletion(patterns_token_df),
        StatisticalReplacement(xor_token_df)
    ]
fallback_augmentors = [
    RandomInsertion(available_tokens), RandomDeletion(), RandomReplacement(available_tokens)
]

applicability_info = check_applicability(train_token_x, main_augmentors, fallback_augmentors)
augmented_data, max_len, aug_type_counts = generate_augmented_views(train_token_x, applicability_info, main_augmentors, fallback_augmentors)

augmented_1_tensor = torch.tensor(np.array([ex['augmented_1'] for ex in augmented_data]), dtype=torch.long)
augmented_2_tensor = torch.tensor(np.array([ex['augmented_2'] for ex in augmented_data]), dtype=torch.long)

dataset = TensorDataset(augmented_1_tensor, augmented_2_tensor)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

f_online = SiamSAEncoder(EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT, max_len, len(x_word_dict), HIDDEN_DIM, FEATURE_DIM).to(device)
h_online = SiamSAPredictor(FEATURE_DIM).to(device)

f_target = SiamSAEncoder(EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT, max_len, len(x_word_dict), HIDDEN_DIM, FEATURE_DIM).to(device)
f_target.load_state_dict(f_online.state_dict()) 
for param in f_target.parameters():
    param.requires_grad = False 

steps_per_epoch = len(dataloader)
total_steps = steps_per_epoch * EPOCHS

optimizer = torch.optim.SGD(list(f_online.parameters()) + list(h_online.parameters()), lr=BASE_LR, momentum=0.9, weight_decay=WEIGHT_DECAY)

def lr_lambda(current_step):
    warmup_steps = WARMUP_EPOCHS * steps_per_epoch
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + np.cos(np.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def update_ema_weights(target_model, online_model, tau):
    with torch.no_grad():
        for target_param, online_param in zip(target_model.parameters(), online_model.parameters()):
            target_param.data = tau * target_param.data + (1.0 - tau) * online_param.data

def byol_loss(p, z):
    p = F.normalize(p, dim=1)
    z = F.normalize(z, dim=1)
    return 2 - 2 * (p * z).sum(dim=-1).mean()

epoch_wise_loss = []
step = 0

for epoch in range(EPOCHS):
    f_online.train()
    h_online.train()
    step_wise_loss = []
    
    for x1, x2 in dataloader:
        x1, x2 = x1.to(device), x2.to(device)
        
        optimizer.zero_grad()
        z1, z2 = f_online(x1), f_online(x2)
        p1, p2 = h_online(z1), h_online(z2)
        
        with torch.no_grad():
            t1, t2 = f_target(x1), f_target(x2)
            
        loss = byol_loss(p1, t2) + byol_loss(p2, t1)
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        tau = 1 - (1 - TAU_BASE) * (np.cos(np.pi * step / total_steps) + 1) / 2
        update_ema_weights(f_target, f_online, tau)
        
        step_wise_loss.append(loss.item())
        step += 1
        
    mean_loss = np.mean(step_wise_loss)
    epoch_wise_loss.append(mean_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {mean_loss:.4f} - Tau: {tau:.5f}")

os.makedirs("PreTrainedModels", exist_ok=True)
model_path = f"PreTrainedModels/{dataName}_pretrained.pth"
torch.save(f_online.state_dict(), model_path)
print(f"Model saved at {model_path}")