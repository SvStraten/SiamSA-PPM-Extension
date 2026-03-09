import os
import argparse
import numpy as np
import pandas as pd
import torch
from sklearn import metrics
from collections import Counter
import time
import random

from Preprocessing.loader import LogsDataLoader
from Preprocessing.processor import LogsDataProcessor

def data_loader_nap(dataName):
    data_processor = LogsDataProcessor(name=dataName, filepath=f"data/{dataName}.csv",
                                        columns=["case:concept:name", "concept:name", "time:timestamp"], 
                                        dir_path='datasets', pool=4)
    return LogsDataLoader(name=dataName)
    
def data_loader_fop(dataName):
    data_processor = LogsDataProcessor(name=dataName, filepath=f"data/{dataName}.csv",
                                        columns=["case:concept:name", "concept:name", "time:timestamp"], 
                                        dir_path='datasets', pool=4)
    return LogsDataLoader(name=dataName)

def preprocess_nap(data):
    (train_df, test_df, x_word_dict, y_word_dict, max_case_length,
        vocab_size, num_output) = data.load_data("next_activity")

    train_token_x, train_token_y = data.prepare_data_next_activity(train_df,
        x_word_dict, y_word_dict, max_case_length)
    
    return train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output, train_token_x, train_token_y

def preprocess_fop(data):
    (train_df, test_df, x_word_dict, y_word_dict, max_case_length,
        vocab_size, num_output) = data.load_data("final_outcome")

    train_token_x, train_token_y = data.prepare_data_final_outcome(train_df,
        x_word_dict, y_word_dict, max_case_length)
    
    return train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output, train_token_x, train_token_y

def preprocess_input(token_x, max_trace_length, number_of_activities):
    x = np.zeros((token_x.shape[0], max_trace_length, number_of_activities, 2))
    
    for i, sequence in enumerate(token_x):
        for t, act_id in enumerate(sequence):
            act_id = int(act_id) 
            if 0 <= act_id < number_of_activities:
                x[i, t, act_id, 0] = 1
                x[i, t, act_id, 1] = t / max_trace_length
    return x

def check_gpu():
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    else:
        print("No GPU found. Using CPU.")
        return torch.device("cpu")

def evaluate_per_k(model, test_df, data, x_word_dict, y_word_dict, max_case_length):
    device = next(model.parameters()).device
    model.eval()
    k, accuracies, fscores, precisions, recalls = [], [], [], [], []

    with torch.no_grad():
        for i in range(max_case_length):
            test_data_subset = test_df[test_df["k"] == i]
            if len(test_data_subset) > 0:
                test_token_x, test_token_y = data.prepare_data_next_activity(
                    test_data_subset, x_word_dict, y_word_dict, max_case_length)

                # PyTorch Inference
                test_tensor_x = torch.tensor(test_token_x, dtype=torch.long).to(device)
                outputs = model(test_tensor_x)
                y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

                accuracy = metrics.accuracy_score(test_token_y, y_pred)
                precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
                    test_token_y, y_pred, average="macro", zero_division=0)

                k.append(i)
                accuracies.append(accuracy)
                fscores.append(fscore)
                precisions.append(precision)
                recalls.append(recall)

    return {
        "k": k, "accuracies": accuracies, "fscores": fscores,
        "precisions": precisions, "recalls": recalls
    }

def evaluate_global(model, test_df, data, x_word_dict, y_word_dict, max_case_length):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_trues = []

    start_time = time.time()

    with torch.no_grad():
        for i in range(max_case_length):
            test_data_subset = test_df[test_df["k"] == i]
            if len(test_data_subset) > 0:
                test_token_x, test_token_y = data.prepare_data_next_activity(
                    test_data_subset, x_word_dict, y_word_dict, max_case_length)

                test_tensor_x = torch.tensor(test_token_x, dtype=torch.long).to(device)
                outputs = model(test_tensor_x)
                y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(y_pred)
                all_trues.extend(test_token_y)

    elapsed_time = time.time() - start_time 

    global_accuracy = metrics.accuracy_score(all_trues, all_preds)
    global_precision, global_recall, global_fscore, _ = metrics.precision_recall_fscore_support(
        all_trues, all_preds, average="weighted", zero_division=0)

    return {
        "global_accuracy": global_accuracy, "global_precision": global_precision,
        "global_recall": global_recall, "global_fscore": global_fscore,
        "inference_time": elapsed_time  
    }

def evaluate_per_k_fop(model, test_df, data, x_word_dict, y_word_dict, max_case_length):
    device = next(model.parameters()).device
    model.eval()
    all_label_results = []
    labels = test_df["final_outcome"].unique()

    with torch.no_grad():
        for label_name in sorted(labels):
            label_subset = test_df[test_df["final_outcome"] == label_name]

            for k in range(max_case_length):
                k_subset = label_subset[label_subset["k"] == k]
                if len(k_subset) == 0:
                    continue

                test_token_x, test_token_y = data.prepare_data_final_outcome(
                    k_subset, x_word_dict, y_word_dict, max_case_length
                )

                test_tensor_x = torch.tensor(test_token_x, dtype=torch.long).to(device)
                outputs = model(test_tensor_x)
                y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

                acc = metrics.accuracy_score(test_token_y, y_pred)
                f1 = metrics.f1_score(test_token_y, y_pred, average="weighted", zero_division=0)

                all_label_results.append({
                    "Label": label_name,
                    "Prefix Length (k)": k,
                    "Accuracy": acc * 100,
                    "F-score": f1 * 100
                })

    return all_label_results

def evaluate_global_fop(model, test_df, data, x_word_dict, y_word_dict, max_case_length):
    device = next(model.parameters()).device
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for i in range(max_case_length + 2):
            test_data_subset = test_df[test_df["k"] == i]
            if len(test_data_subset) > 0:
                test_token_x, test_token_y = data.prepare_data_final_outcome(
                    test_data_subset, x_word_dict, y_word_dict, max_case_length)

                test_tensor_x = torch.tensor(test_token_x, dtype=torch.long).to(device)
                outputs = model(test_tensor_x)
                y_pred = torch.argmax(outputs, dim=1).cpu().numpy()

                all_preds.extend(y_pred)
                all_trues.extend(test_token_y)

    global_accuracy = metrics.accuracy_score(all_trues, all_preds)
    global_precision, global_recall, global_fscore, _ = metrics.precision_recall_fscore_support(
        all_trues, all_preds, average="weighted", zero_division=0)

    index_to_label = {v: k for k, v in y_word_dict.items()}
    y_true_labels = [index_to_label[i] for i in all_trues]
    y_pred_labels = [index_to_label[i] for i in all_preds]
    unique_labels = sorted(set(y_true_labels))
    
    label_counts = Counter(y_true_labels)
    per_label_results = {}
    
    for label in unique_labels:
        y_true_bin = [1 if y == label else 0 for y in y_true_labels]
        y_pred_bin = [1 if y == label else 0 for y in y_pred_labels]

        acc = metrics.accuracy_score(y_true_bin, y_pred_bin)
        f1 = metrics.f1_score(y_true_bin, y_pred_bin, zero_division=0)

        per_label_results[label] = {
            "accuracy": acc,
            "f1_score": f1,
            "count": label_counts[label]
        }

    return {
        "global_accuracy": global_accuracy, "global_precision": global_precision,
        "global_recall": global_recall, "global_fscore": global_fscore,
        "per_label_results": per_label_results
    }
    
def check_applicability(train_token_x, main_augmentors, fallback_augmentors):
    results = []

    for idx, sequence in enumerate(train_token_x):
        sequence = list(sequence)
        while sequence and sequence[-1] == 0:
            sequence.pop()

        applicable_main = [aug.get_name() for aug in main_augmentors if aug.is_applicable(sequence)]
        applicable_fallback = [aug.get_name() for aug in fallback_augmentors if aug.is_applicable(sequence)]

        results.append({
            "index": idx,
            "applicable_main": applicable_main,
            "applicable_fallback": applicable_fallback
        })
    return results

def generate_augmented_views(train_token_x, applicability_info, main_augmentors, fallback_augmentors, max_attempts=30):
    augmented_dataset = []
    augmentor_usage = Counter()
    temp_sequences = []

    def apply_random_augmentor(pool, sequence, exclude_names=None):
        exclude_names = exclude_names or set()
        usable_pool = [aug for aug in pool if aug.get_name() not in exclude_names and aug.is_applicable(sequence)]
        if not usable_pool:
            return None, None
        aug = random.choice(usable_pool)
        return aug.augment(sequence), aug.get_name()

    def try_second_augmentor(sequence, aug1, pool_main, pool_fallback, max_attempts):
        attempts = 0
        combined_pool = [aug for aug in (pool_main + pool_fallback) if aug.is_applicable(sequence)]
        if not combined_pool:
            return None, None

        while attempts < max_attempts:
            aug = random.choice(combined_pool)
            aug2_candidate = aug.augment(sequence)
            name2_candidate = aug.get_name()

            if aug2_candidate != aug1:
                return aug2_candidate, name2_candidate
            attempts += 1
        return None, None

    for item in applicability_info:
        idx = item["index"]
        sequence = list(train_token_x[idx])
        sequence = [x for x in sequence if x != 0]

        applicable_main = [aug for aug in main_augmentors if aug.get_name() in item["applicable_main"]]
        applicable_fallback = [aug for aug in fallback_augmentors if aug.get_name() in item["applicable_fallback"]]

        if applicable_main:
            pool = applicable_main
        elif applicable_fallback:
            pool = applicable_fallback
        else:
            raise ValueError(f"No applicable augmentors for sequence at index {idx}")

        aug1, name1 = apply_random_augmentor(pool, sequence)
        if aug1 is None:
            raise ValueError(f"No usable augmentors found for first augmentation at index {idx}")

        aug2, name2 = try_second_augmentor(sequence, aug1, applicable_main, applicable_fallback, max_attempts)

        if aug2 is None:
            print(f"[DEBUG] Index {idx} - Original: {sequence}")
            print(f"[DEBUG] Augmentor 1: {name1}, Output: {aug1}")
            raise ValueError(f"Failed to generate a distinct second augmented view at index {idx} after {max_attempts} attempts.")

        augmentor_usage[name1] += 1
        augmentor_usage[name2] += 1

        temp_sequences.append({
            "original": sequence,
            "augmented_1": aug1,
            "augmented_2": aug2,
            "augmentor_1": name1,
            "augmentor_2": name2
        })

    max_len = max(max(len(seq["original"]), len(seq["augmented_1"]), len(seq["augmented_2"])) for seq in temp_sequences)

    def left_pad(seq, target_len):
        return [0.0] * (target_len - len(seq)) + seq

    for item in temp_sequences:
        augmented_dataset.append({
            "original": np.array(left_pad(item["original"], max_len), dtype=np.float32),
            "augmented_1": np.array(left_pad(item["augmented_1"], max_len), dtype=np.float32),
            "augmented_2": np.array(left_pad(item["augmented_2"], max_len), dtype=np.float32),
            "augmentor_1": item["augmentor_1"],
            "augmentor_2": item["augmentor_2"]
        })

    return augmented_dataset, max_len, augmentor_usage