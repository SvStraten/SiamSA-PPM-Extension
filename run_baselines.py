import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter
import warnings

# Suppress warnings for clean output
warnings.filterwarnings("ignore")

from Preprocessing.utils import data_loader_nap, preprocess_nap, check_gpu
from Preprocessing.loader import pad_sequences_pre
from Augmentation.kappel_augmentation import apply_kappel_eda
from Augmentation.augmentation_module import (
    get_patterns, map_patterns_to_tokens,
    get_xor_candidates, map_xor_candidates_to_tokens,
    StatisticalInsertion, StatisticalDeletion, StatisticalReplacement
)

# Import Baselines
from Baselines.tax_lstm import TaxLSTM
from Baselines.dimauro_cnn import DiMauroIDCNN
from Baselines.pasquadibisceglie_cnn import Pasquadibisceglie2DCNN
from Baselines.bukhsh_transformer import BukhshTransformer

# Configuration
DATASET = 'bpic13_o'
AUG_FACTOR = 1.2
BATCH_SIZE = 256
EPOCHS = 30  # Adjust as needed
DEVICE = check_gpu()

def apply_our_eda(train_x, train_y, x_word_dict, dataName, augmentation_factor=1.2, max_attempts=10):
    """
    Applies 'Our Augmentation' (SiamSA-PPM's combi strategy) for offline baseline training.
    """
    print(f"\n[Our EDA] Extracting Statistical Patterns for {dataName}...")
    patterns_df = get_patterns(f'datasets/{dataName}/processed/next_activity_train.csv')
    patterns_token_df = map_patterns_to_tokens(patterns_df, x_word_dict)

    xor_df = get_xor_candidates(f'datasets/{dataName}/processed/next_activity_train.csv')
    xor_token_df = map_xor_candidates_to_tokens(xor_df, x_word_dict)

    our_augmentors = [
        StatisticalInsertion(patterns_token_df),
        StatisticalDeletion(patterns_token_df),
        StatisticalReplacement(xor_token_df)
    ]

    num_original = len(train_x)
    num_to_generate = int(np.ceil(augmentation_factor * num_original)) - num_original
    
    if num_to_generate <= 0: return train_x, train_y

    augmented_x, augmented_y = [], []
    indices = list(range(num_original))
    usage = Counter()
    
    print(f"[Our EDA] Generating {num_to_generate} synthetic prefixes (Factor: {augmentation_factor}x)...")
    for _ in range(num_to_generate):
        idx = random.choice(indices)
        sequence, label = list(train_x[idx]), train_y[idx]
        
        applicable_augmentors = [aug for aug in our_augmentors if aug.is_applicable(sequence)]
        if not applicable_augmentors:
            augmented_x.append(sequence)
            augmented_y.append(label)
            usage["Duplication (No Augment)"] += 1
            continue
            
        success = False
        for _ in range(max_attempts):
            aug = random.choice(applicable_augmentors)
            new_sequence = aug.augment(sequence)
            if new_sequence != sequence:
                augmented_x.append(new_sequence)
                augmented_y.append(label)
                usage[aug.get_name()] += 1
                success = True
                break
                
        if not success:
            augmented_x.append(sequence)
            augmented_y.append(label)
            usage["Duplication (No Augment)"] += 1

    final_augmented_x = np.array(augmented_x, dtype=np.float32)
    final_augmented_y = np.array(augmented_y, dtype=np.float32)
    
    combined_x = np.concatenate([train_x, final_augmented_x], axis=0)
    combined_y = np.concatenate([train_y, final_augmented_y], axis=0)
    
    shuffled_indices = np.random.permutation(len(combined_x))
    return combined_x[shuffled_indices], combined_y[shuffled_indices]

def train_and_evaluate_baseline(model, train_x, train_y, test_x, test_y, num_activities, is_pasquad=False):
    """Trains a baseline model safely on GPU using batches and returns Accuracy and F1-score."""
    model.to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    train_x_tensor = torch.tensor(train_x, dtype=torch.long)
    test_x_tensor = torch.tensor(test_x, dtype=torch.long)
    train_y_tensor = torch.tensor(train_y, dtype=torch.long)
    test_y_tensor = torch.tensor(test_y, dtype=torch.long)
        
    train_ds = TensorDataset(train_x_tensor, train_y_tensor)
    test_ds = TensorDataset(test_x_tensor, test_y_tensor)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # Training Loop
    model.train()
    for epoch in range(EPOCHS):
        for x_batch, y_batch in train_loader:
            # Transform to 2D image map dynamically to save RAM
            if is_pasquad:
                x_batch = F.one_hot(x_batch, num_classes=num_activities+1).float().unsqueeze(1)
                
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
    # Evaluation Loop (Batched to prevent CUDA OOM)
    model.eval()
    all_preds = []
    with torch.no_grad():
        for x_batch, _ in test_loader:
            if is_pasquad:
                x_batch = F.one_hot(x_batch, num_classes=num_activities+1).float().unsqueeze(1)
                
            x_batch = x_batch.to(DEVICE)
            outputs = model(x_batch)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            
    acc = accuracy_score(test_y, all_preds)
    f1 = f1_score(test_y, all_preds, average='macro', zero_division=0)
    return acc * 100, f1 * 100

def main():
    print(f"===========================================================")
    print(f" Starting Baseline Evaluation for Table 2: {DATASET.upper()}")
    print(f" Executing on: {DEVICE}")
    print(f"===========================================================")
    
    # 1. Load Original Data
    data = data_loader_nap(DATASET)
    train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output, train_x, train_y = preprocess_nap(data)
    
    # Extract Test Set
    test_token_x, test_token_y = data.prepare_data_next_activity(test_df, x_word_dict, y_word_dict, max_case_length, shuffle=False)
    test_x_padded = pad_sequences_pre(test_token_x, maxlen=max_case_length)

    # 2. Generate Kappel Augmented Training Data
    print("\n---> Generating Kappel EDA Data")
    train_x_kappel, train_y_kappel = apply_kappel_eda(train_x, train_y, x_word_dict, augmentation_factor=AUG_FACTOR)
    train_x_kappel_padded = pad_sequences_pre(train_x_kappel, maxlen=max_case_length)

    # 3. Generate Our Augmented Training Data
    print("\n---> Generating Our EDA Data")
    train_x_ours, train_y_ours = apply_our_eda(train_x, train_y, x_word_dict, DATASET, augmentation_factor=AUG_FACTOR)
    train_x_ours_padded = pad_sequences_pre(train_x_ours, maxlen=max_case_length)

    results = []

    baselines = {
        "Tax (LSTM)": lambda: TaxLSTM(num_output, max_case_length),
        "Di Mauro (ID-CNN)": lambda: DiMauroIDCNN(num_output, max_case_length),
        "Pasquadibisceglie (2D-CNN)": lambda: Pasquadibisceglie2DCNN(num_output),
        "Bukhsh (Transformer)": lambda: BukhshTransformer(num_output, max_case_length)
    }

    # 4. Train and Evaluate
    for name, model_init in baselines.items():
        is_pasquad = "Pasquadibisceglie" in name
        
        # --- Evaluate Kappel ---
        print(f"\nTraining {name} with Kappel Augmentation...")
        model_kappel = model_init()
        acc_kappel, f1_kappel = train_and_evaluate_baseline(
            model_kappel, train_x_kappel_padded, train_y_kappel, test_x_padded, test_token_y, vocab_size, is_pasquad
        )
        
        # --- Evaluate Ours ---
        print(f"Training {name} with Our Augmentation...")
        model_ours = model_init()
        acc_ours, f1_ours = train_and_evaluate_baseline(
            model_ours, train_x_ours_padded, train_y_ours, test_x_padded, test_token_y, vocab_size, is_pasquad
        )

        results.append({
            "Baseline": name,
            "Kappel Acc": f"{acc_kappel:.2f}",
            "Kappel F1": f"{f1_kappel:.2f}",
            "Our Aug Acc": f"{acc_ours:.2f}",
            "Our Aug F1": f"{f1_ours:.2f}"
        })

    # 5. Print Final Table 2 Results
    print("\n\n================ FINAL TABLE 2 RESULTS ================")
    results_df = pd.DataFrame(results)
    print(results_df.to_markdown(index=False))
    print("=======================================================")

if __name__ == "__main__":
    main()