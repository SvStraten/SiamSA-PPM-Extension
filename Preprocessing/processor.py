import os
import json
import pandas as pd
import numpy as np
import datetime
from multiprocessing import Pool

class LogsDataProcessor:
    def __init__(self, name, filepath, columns, dir_path="./datasets/processed", pool=1):
        """Provides support for processing raw logs."""
        self._name = name
        self._filepath = filepath
        self._org_columns = columns
        self._dir_path = dir_path
        if not os.path.exists(f"{dir_path}/{self._name}/processed"):
            os.makedirs(f"{dir_path}/{self._name}/processed")
        self._dir_path = f"{self._dir_path}/{self._name}/processed"
        self._pool = pool

    def _load_df(self, sort_temporally=False):
        df = pd.read_csv(self._filepath)
        df = df[self._org_columns]
        df.columns = ["case:concept:name", "concept:name", "time:timestamp"]
        df["concept:name"] = df["concept:name"].str.lower()
        df["concept:name"] = df["concept:name"].str.replace(" ", "-")
        df["time:timestamp"] = pd.to_datetime(df["time:timestamp"], utc=True, errors='coerce')
        if sort_temporally:
            df.sort_values(by=["time:timestamp"], inplace=True)
        return df

    def _extract_logs_metadata(self, df):
        keys = ["[PAD]", "[UNK]"]
        activities = list(df["concept:name"].unique())
        keys.extend(activities)
        val = range(len(keys))

        coded_activity = dict({"x_word_dict": dict(zip(keys, val))})
        code_activity_normal = dict({"y_word_dict": dict(zip(activities, range(len(activities))))})

        coded_activity.update(code_activity_normal)
        coded_json = json.dumps(coded_activity)
        with open(f"{self._dir_path}/metadata.json", "w") as metadata_file:
            metadata_file.write(coded_json)

    def _next_activity_helper_func(self, df):
        case_id, case_name = "case:concept:name", "concept:name"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "next_act"])
        idx = 0
        unique_cases = df[case_id].unique()
        for _, case in enumerate(unique_cases):
            act = df[df[case_id] == case][case_name].to_list()
            for i in range(1, len(act) - 1):
                prefix = np.where(i == 0, act[0], " ".join(act[:i+1]))        
                next_act = act[i+1]
                processed_df.at[idx, "case_id"]  = case
                processed_df.at[idx, "prefix"]   = prefix
                processed_df.at[idx, "k"]        = i
                processed_df.at[idx, "next_act"] = next_act
                idx += 1
        return processed_df
    
    def _final_outcome_helper_func(self, df, dataName):
        case_id_col, activity_col = "case:concept:name", "concept:name"
        processed_df = pd.DataFrame(columns=["case_id", "prefix", "k", "final_outcome"])
        idx = 0
        if dataName == 'bpic12':
            outcome_labels = {"a_approved", "a_declined", "a_cancelled"}
        elif dataName == 'sepsis':
            outcome_labels = {"a_cancelled", "release-b", "release-c", "release-d", "release-e"}
        else:
            outcome_labels = set()
            
        unique_cases = df[case_id_col].unique()

        for case in unique_cases:
            case_df = df[df[case_id_col] == case]
            activities = case_df[activity_col].tolist()
            final_outcome = next((act for act in reversed(activities) if act in outcome_labels), None)

            if final_outcome is None:
                continue

            for i in range(1, len(activities)):
                prefix = " ".join(activities[:i+1])
                processed_df.at[idx, "case_id"] = case
                processed_df.at[idx, "prefix"] = prefix
                processed_df.at[idx, "k"] = i
                processed_df.at[idx, "final_outcome"] = final_outcome
                idx += 1

        return processed_df

    def _process_next_activity(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._next_activity_helper_func, df_split))
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        train_df.to_csv(f"{self._dir_path}/next_activity_train.csv", index=False)
        test_df.to_csv(f"{self._dir_path}/next_activity_test.csv", index=False)
        
    def _process_final_outcome(self, df, train_list, test_list):
        df_split = np.array_split(df, self._pool)
        with Pool(processes=self._pool) as pool:
            processed_df = pd.concat(pool.imap_unordered(self._final_outcome_helper_func, df_split))
        
        train_df = processed_df[processed_df["case_id"].isin(train_list)]
        test_df = processed_df[processed_df["case_id"].isin(test_list)]
        
        train_df.to_csv(f"{self._dir_path}/final_outcome_train.csv", index=False)
        test_df.to_csv(f"{self._dir_path}/final_outcome_test.csv", index=False)

    def process_logs(self, task, sort_temporally=False, train_test_ratio=0.80):
        df = self._load_df(sort_temporally)
        self._extract_logs_metadata(df)
        train_test_ratio = int(abs(df["case:concept:name"].nunique() * train_test_ratio))
        train_list = df["case:concept:name"].unique()[:train_test_ratio]
        test_list = df["case:concept:name"].unique()[train_test_ratio:]
        if task == "next_activity":
            self._process_next_activity(df, train_list, test_list)
        elif task == "final_outcome":
            self._process_final_outcome(df, train_list, test_list)
        else:
            raise ValueError("Invalid task.")