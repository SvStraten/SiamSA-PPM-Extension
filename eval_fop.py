import pandas as pd
import argparse
import torch
import warnings
import os

from Preprocessing.utils import data_loader_fop, preprocess_fop, check_gpu, evaluate_per_k_fop, evaluate_global_fop
from Model.model import SiamSAEncoder, DownstreamClassifier # Updated Import

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Configure experiment settings.')
parser.add_argument('--dataName', type=str, default='sepsis', help='Name of the dataset')
args = parser.parse_args()

dataName = args.dataName
repetitions = 1
EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT = 128, 4, 256, 2, 0.2
HIDDEN_DIM, FEATURE_DIM = 256, 256

print(f"\n📊 Evaluating dataset: {dataName}")

data = data_loader_fop(dataName)
train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, num_output, train_token_x, train_token_y = preprocess_fop(data)

device = check_gpu()
all_k_results = []
all_results = []
method_summaries = []
dataset_summaries = {}

for rep in range(repetitions):
    model_path = f"FOPModels/{dataName}_fop_{rep}.pth"
    
    encoder = SiamSAEncoder(EMBED_DIM, NUM_HEADS, FF_DIM, LAYERS, DROPOUT, max_case_length, vocab_size, HIDDEN_DIM, FEATURE_DIM)
    model = DownstreamClassifier(encoder, EMBED_DIM, len(y_word_dict))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    k_results = evaluate_per_k_fop(model, test_df, data, x_word_dict, y_word_dict, max_case_length)

    for entry in k_results:
        entry["Repetition"] = rep
    all_k_results.extend(k_results)

    df = pd.DataFrame(all_k_results)
    summary = df.groupby(["Label", "Prefix Length (k)"]).agg({"Accuracy": "mean", "F-score": "mean"}).reset_index()
    method_summaries.append(summary)

    if method_summaries:
        combined_summary = pd.concat(method_summaries, ignore_index=True)
        dataset_summaries[dataName] = combined_summary
        print(f"✅ Completed summary for dataset: {dataName}")