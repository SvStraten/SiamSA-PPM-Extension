import os
import json
import numpy as np
import pandas as pd
from sklearn import utils

def pad_sequences_pre(sequences, maxlen, value=0.0):
    """
    Pure NumPy implementation of Keras's pre-padding.
    """
    padded = np.full((len(sequences), maxlen), value, dtype=np.float32)
    for i, seq in enumerate(sequences):
        trunc = seq[-maxlen:] # Truncate from the left if it's too long
        padded[i, -len(trunc):] = trunc # Pad on the left
    return padded

class LogsDataLoader:
    def __init__(self, name, dir_path="./datasets"):
        self._dir_path = f"{dir_path}/{name}/processed"

    def prepare_data_next_activity(self, df, x_word_dict, y_word_dict, max_case_length, shuffle=True):
        x = df["prefix"].values
        y = df["next_act"].values
        if shuffle:
            x, y = utils.shuffle(x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        token_y = list()
        for _y in y:
            token_y.append(y_word_dict[_y])

        token_x = pad_sequences_pre(token_x, maxlen=max_case_length)
        token_x = np.array(token_x, dtype=np.float32)
        token_y = np.array(token_y, dtype=np.float32)

        return token_x, token_y
    
    def prepare_data_final_outcome(self, df, x_word_dict, y_word_dict, max_case_length, shuffle=True):
        x = df["prefix"].values
        y = df["final_outcome"].values
        if shuffle:
            x, y = utils.shuffle(x, y)

        token_x = list()
        for _x in x:
            token_x.append([x_word_dict[s] for s in _x.split()])

        token_y = list()
        for _y in y:
            token_y.append(y_word_dict[_y])

        token_x = pad_sequences_pre(token_x, maxlen=max_case_length)
        token_x = np.array(token_x, dtype=np.float32)
        token_y = np.array(token_y, dtype=np.float32)

        return token_x, token_y

    def get_max_case_length(self, train_x):
        train_token_x = list()
        for _x in train_x:
            train_token_x.append(len(_x.split()))
        return max(train_token_x)

    def load_data(self, task):
        train_df = pd.read_csv(f"{self._dir_path}/{task}_train.csv")
        test_df = pd.read_csv(f"{self._dir_path}/{task}_test.csv")

        with open(f"{self._dir_path}/metadata.json", "r") as json_file:
            metadata = json.load(json_file)

        x_word_dict = metadata["x_word_dict"]
        y_word_dict = metadata["y_word_dict"]
        max_case_length = self.get_max_case_length(train_df["prefix"].values)
        vocab_size = len(x_word_dict) 
        total_classes = len(y_word_dict)

        return (train_df, test_df, x_word_dict, y_word_dict, max_case_length, vocab_size, total_classes)