import pandas as pd
import argparse
import torch
import warnings
import os

from Preprocessing.utils import data_loader_nap, preprocess_nap, check_gpu, evaluate_per_k, evaluate_global
from Model.model import SiamSAEncoder, DownstreamClassifier # Updated Import

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Configure experiment settings.')
parser.add_argument('--dataName', type=str, default='bpic13_o', help='Name of the dataset')
args = parser.parse_args()

dataName = args.dataName
repetitions = 1
EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT = 128, 4, 256, 2, 0.2
HIDDEN_DIM, FEATURE_DIM = 256, 256

print(f"\n📊 Evaluating dataset: {dataName}")

data = data_loader_nap(dataName)
train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output, train_token_x, train_token_y = preprocess_nap(data)

device = check_gpu()
all_k_results = []
all_results = []
method_summaries = []

for rep in range(repetitions):
    model_path = f"NAPModels/{dataName}_nap_{rep}.pth"
    
    encoder = SiamSAEncoder(EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT, max_case_length, vocab_size, HIDDEN_DIM, FEATURE_DIM)
    model = DownstreamClassifier(encoder, EMBED_DIM, len(y_word_dict))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    k_results = evaluate_per_k(model, test_df, data, x_word_dict, y_word_dict, max_case_length)
    global_results = evaluate_global(model, test_df, data, x_word_dict, y_word_dict, max_case_length)

    all_k_results.append(k_results)

    all_results.append({
        "Dataset": dataName,
        "Repetition": rep,
        "Test Accuracy (%)": global_results['global_accuracy'] * 100,
        "Macro F1-score (%)": global_results['global_fscore'] * 100,
        "Inference Time (s)": global_results['inference_time']
    })
        
    records = []
    for rep_id, result in enumerate(all_k_results):
        for k_val, acc, f1 in zip(result["k"], result["accuracies"], result["fscores"]):
            records.append({
                "Repetition": rep_id, "Prefix Length (k)": k_val, "Accuracy": acc * 100, "F-score": f1 * 100,
            })

    df = pd.DataFrame(records)
    summary = df.groupby(["Prefix Length (k)"]).agg({"Accuracy": "mean", "F-score": "mean"}).reset_index()
    method_summaries.append(summary)
    
    summary_df = pd.DataFrame(all_results)
    grouped = summary_df.groupby(["Dataset"]).agg({
        "Test Accuracy (%)": ["mean", "std"],
        "Macro F1-score (%)": ["mean", "std"],
        "Inference Time (s)": ["mean", "std"]
    }).reset_index()

    grouped.columns = ['Dataset', 'Acc Mean', 'Acc Std', 'F1 Mean', 'F1 Std', 'Time Mean (s)', 'Time Std (s)']
    grouped = grouped.round(2)

    print("\n📋 Final Summary Table:")
    print(grouped)