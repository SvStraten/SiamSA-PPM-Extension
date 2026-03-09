import numpy as np
import random
from collections import Counter

# Import the standard random augmentors utilized by Kappel & Jablonski's EDA
from Augmentation.augmentation_module import RandomInsertion, RandomDeletion, RandomReplacement, RandomSwap

def apply_kappel_eda(train_x, train_y, x_word_dict, augmentation_factor=1.2, max_attempts=10):
    """
    Applies the Kappel and Jablonski Easy Data Augmentation (EDA) strategy.
    Randomly applies Insertion, Deletion, Replacement, and Swap to generate 
    a larger, synthetic dataset for offline supervised training.
    """
    available_tokens = list(x_word_dict.values())
    
    # Initialize Kappel's 4 random EDA augmentors
    kappel_augmentors = [
        RandomInsertion(available_tokens),
        RandomDeletion(),
        RandomReplacement(available_tokens),
        RandomSwap()
    ]
    
    num_original = len(train_x)
    num_to_generate = int(np.ceil(augmentation_factor * num_original)) - num_original
    
    if num_to_generate <= 0:
        print("[Kappel EDA] Augmentation factor <= 1.0. No synthetic traces generated.")
        return train_x, train_y

    augmented_x = []
    augmented_y = []
    
    indices = list(range(num_original))
    augmentor_usage = Counter()
    
    print(f"\n[Kappel EDA] Generating {num_to_generate} synthetic prefixes (Factor: {augmentation_factor}x)...")
    
    for i in range(num_to_generate):
        # 1. Randomly pick an original prefix to augment
        idx = random.choice(indices)
        sequence = list(train_x[idx])
        label = train_y[idx]
        
        # 2. Check which Kappel augmentors are applicable to this sequence length
        applicable_augmentors = [aug for aug in kappel_augmentors if aug.is_applicable(sequence)]
        
        if not applicable_augmentors:
            # Fallback: duplicate if too short to augment
            augmented_x.append(sequence)
            augmented_y.append(label)
            augmentor_usage["Duplication (No Augment)"] += 1
            continue
            
        # 3. Try generating a unique synthetic trace
        success = False
        for _ in range(max_attempts):
            aug = random.choice(applicable_augmentors)
            new_sequence = aug.augment(sequence)
            
            if new_sequence != sequence:
                augmented_x.append(new_sequence)
                augmented_y.append(label)
                augmentor_usage[aug.get_name()] += 1
                success = True
                break
                
        if not success:
            augmented_x.append(sequence)
            augmented_y.append(label)
            augmentor_usage["Duplication (No Augment)"] += 1

    # Convert generated traces back to PyTorch-compatible NumPy arrays
    final_augmented_x = np.array(augmented_x, dtype=np.float32)
    final_augmented_y = np.array(augmented_y, dtype=np.float32)
    
    print("[Kappel EDA] Augmentor usage statistics:", dict(augmentor_usage))
    
    # 4. Concatenate the original dataset with the new synthetic data
    combined_x = np.concatenate([train_x, final_augmented_x], axis=0)
    combined_y = np.concatenate([train_y, final_augmented_y], axis=0)
    
    # 5. Shuffle the expanded dataset
    shuffled_indices = np.random.permutation(len(combined_x))
    return combined_x[shuffled_indices], combined_y[shuffled_indices]