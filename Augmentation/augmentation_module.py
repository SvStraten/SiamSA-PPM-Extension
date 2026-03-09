import os
import json
import math
import random
import typing
import dataclasses
import abc
from copy import deepcopy
from collections import Counter, defaultdict
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas.core.groupby as pd_groupby
import torch

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.log.util import dataframe_utils
from pm4py.objects.conversion.log import converter as log_converter

# ==============================================================================
# PIPELINE STUBS 
# (To replace the missing imports from augmentation.pipelines...)
# ==============================================================================
class AbstractPipelineContext:
    def get_full_event_log(self):
        pass

class BasePipelineStep:
    pass

# ==============================================================================
# EVENT LOG UTILITIES (From event_log_utils.py)
# ==============================================================================
def df_number_of_traces(df) -> int:
    return df['case:concept:name'].nunique()

def df_average_trace_length(df):
    trace_lengths = df.groupby('case:concept:name').size()
    return trace_lengths.mean()

def df_min_trace_length(df):
    trace_lengths = df.groupby('case:concept:name').size()
    return trace_lengths.min()

def df_max_trace_length(df):
    trace_lengths = df.groupby('case:concept:name').size()
    return trace_lengths.max()

def df_case_durations(df):
    df['time:timestamp'] = pd.to_datetime(df['time:timestamp'])
    start_times = df.groupby('case:concept:name')['time:timestamp'].min()
    end_times = df.groupby('case:concept:name')['time:timestamp'].max()
    durations = end_times - start_times
    return durations

def df_minimal_case_duration(df):
    return df_case_durations(df).min()

def df_maximal_case_duration(df):
    return df_case_durations(df).max()

def df_average_case_duration(df):
    return df_case_durations(df).mean()

def get_activity_resources(df: pd.DataFrame) -> typing.Dict[str, typing.List[str]]:
    if 'concept:name' not in df.columns or 'org:resource' not in df.columns:
        raise ValueError("DataFrame must contain 'concept:name' and 'org:resource' columns")
    activities_resources = {activity: set() for activity in df['concept:name'].unique()}
    for _, event in df.iterrows():
        activities_resources[event['concept:name']].add(event['org:resource'])
    return {activity: list(resources) for activity, resources in activities_resources.items()}

def convert_to_event_log(df):
    df = dataframe_utils.convert_timestamp_columns_in_df(df)
    parameters = {log_converter.Variants.TO_EVENT_LOG.value.Parameters.CASE_ID_KEY: 'case_id'}
    return log_converter.apply(df, parameters=parameters, variant=log_converter.Variants.TO_EVENT_LOG)

def average_trace_length(event_log):
    lengths = [len(trace) for trace in event_log]
    return sum(lengths) / len(lengths)

def min_trace_length(event_log):
    return min(len(trace) for trace in event_log)

def max_trace_length(event_log):
    return max(len(trace) for trace in event_log)

def case_durations(event_log):
    durations = []
    for trace in event_log:
        start_time = trace[0]["time:timestamp"]
        end_time = trace[-1]["time:timestamp"]
        durations.append(end_time - start_time)
    return durations

def minimal_case_duration(event_log):
    return min(case_durations(event_log))

def maximal_case_duration(event_log):
    return max(case_durations(event_log))

def average_case_duration(event_log):
    durations = case_durations(event_log)
    return sum(durations, timedelta()) / len(durations)

def get_case_ids(event_log: typing.Union[EventLog, pd.DataFrame]) -> typing.Set:
    if isinstance(event_log, EventLog):
        return set([trace.attributes['concept:name'] for trace in event_log])
    else:
        return set([s for s in event_log['case:concept:name'].unique()])

def get_activities(event_log: pd.DataFrame) -> typing.Set[str]:
    return set([s for s in event_log['concept:name'].unique()])

def get_resources(event_log: pd.DataFrame) -> typing.Set[str]:
    if 'org:resource' in event_log.columns:
        return set(event_log['org:resource'].unique())
    else:
        return set()

def get_traces(event_log: pd.DataFrame) -> pd_groupby.DataFrameGroupBy:
    return event_log.sort_values(by='time:timestamp').groupby(by='case:concept:name')

def sample_traces(event_log: pd.DataFrame, num_samples: int, strategy: str) -> typing.List[pd.DataFrame]:
    assert strategy in ['uniform', 'rarity', 'first_n']
    cases = event_log['case:concept:name'].unique()
    if len(cases.tolist()) < num_samples:
        raise ValueError(f'Sampling {num_samples} traces, but log only contains {len(cases.tolist())}')
    
    if strategy == 'uniform':
        cases = np.random.choice(cases, num_samples, replace=False)
    elif strategy == 'first_n':
        cases = cases[:num_samples]
    else:
        trace_variants = {}
        for case in cases:
            trace = event_log[event_log['case:concept:name'] == case]
            trace_id = ' >> '.join([e for e in trace['concept:name']])
            if trace_id not in trace_variants:
                trace_variants[trace_id] = []
            trace_variants[trace_id].append(case)
        
        traces_count = [len(traces) for traces in trace_variants.values()]
        total_count = sum(traces_count)
        p = [total_count / v for v in traces_count]
        p = [v / sum(p) for v in p]
        
        selected_trace_variants = np.random.choice(list(trace_variants.keys()), num_samples, replace=True, p=p)
        cases = [random.choice(trace_variants[tv_id]) for tv_id in selected_trace_variants]
        
    return [event_log[event_log['case:concept:name'] == c] for c in cases]

def get_trace_length(trace: pd.DataFrame) -> int:
    return len(trace.index)

def get_variants_as_list(event_log: pd.DataFrame):
    df = event_log.copy()
    df['@@index_in_trace'] = df.groupby('case:concept:name').cumcount()
    variants = []
    for _, df_group in df.sort_values(['case:concept:name', '@@index_in_trace']).groupby('case:concept:name'):
        variants.append([event['concept:name'] for _, event in df_group.iterrows()])
    
    seen = set()
    unique_list_of_lists = []
    for sublist in variants:
        tup = tuple(sublist)
        if tup not in seen:
            seen.add(tup)
            unique_list_of_lists.append(sublist)
    return unique_list_of_lists

def only_control_flow(trace) -> str:
    return ' '.join([event['concept:name'] + "|" for event in trace])

# ==============================================================================
# PATTERN & XOR MINING (From get_patterns.py & get_replacement.py)
# ==============================================================================
def expand_prefix_csv_to_log(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    records = []
    for _, row in df.iterrows():
        case_id = row["case_id"]
        for pos, act in enumerate(row["prefix"].split()):
            records.append({"case:concept:name": case_id, "concept:name": act, "position": pos})
    return pd.DataFrame(records)

def get_significant_activities(log: pd.DataFrame, threshold_ratio: float = 0.00001) -> pd.DataFrame:
    total_cases = log['case:concept:name'].nunique()
    counts = log.groupby('concept:name')['case:concept:name'].nunique()
    significant = counts[counts >= threshold_ratio * total_cases].index
    return log[log['concept:name'].isin(significant)].copy()

def get_significant_transitions(log: pd.DataFrame, threshold: float = 0.00001) -> pd.DataFrame:
    transition_counts = Counter()
    for _, trace in log.groupby('case:concept:name'):
        events = trace.sort_values("position")["concept:name"].tolist()
        for i in range(len(events) - 1):
            transition_counts[(events[i], events[i + 1])] += 1

    total = sum(transition_counts.values())
    df = pd.DataFrame([
        {'Source': s, 'Target': t, 'Probability': c / total}
        for (s, t), c in transition_counts.items()
    ])
    return df[df['Probability'] >= threshold]

def get_significant_paths(log: pd.DataFrame, threshold: float = 0.001, max_length: int = 3) -> pd.DataFrame:
    path_counts = defaultdict(int)
    for _, trace in log.groupby('case:concept:name'):
        events = trace.sort_values("position")["concept:name"].tolist()
        for i in range(len(events)):
            for j in range(i + 1, min(i + max_length + 1, len(events) + 1)):
                path_counts[tuple(events[i:j])] += 1

    total = sum(path_counts.values())
    df = pd.DataFrame([
        {'Path': ' -> '.join(p), 'Frequency': c, 'Support': c / total}
        for p, c in path_counts.items()
    ])
    return df[df['Support'] >= threshold]

def get_patterns(csv_path: str, transition_threshold=0.2, path_threshold=0.2, max_path_length=3, activity_threshold=0.00001) -> pd.DataFrame:
    log = get_significant_activities(expand_prefix_csv_to_log(csv_path), activity_threshold)
    transitions = get_significant_transitions(log, transition_threshold)
    paths = get_significant_paths(log, path_threshold, max_path_length)

    patterns = []
    for _, row in transitions.iterrows():
        patterns.append({'Starting Activity': row['Source'], 'Ending Activity': row['Target'], 'Intermediate Activities': []})

    max_intermediates = 0
    for _, row in paths.iterrows():
        path = row['Path'].split(' -> ')
        intermediates = path[1:-1] if len(path) > 2 else []
        max_intermediates = max(max_intermediates, len(intermediates))
        patterns.append({'Starting Activity': path[0], 'Ending Activity': path[-1], 'Intermediate Activities': intermediates})

    records = []
    for p in patterns:
        rec = {'Starting Activity': p['Starting Activity'], 'Ending Activity': p['Ending Activity']}
        for i in range(max_intermediates):
            rec[f'Intermediate {i + 1}'] = p['Intermediate Activities'][i] if i < len(p['Intermediate Activities']) else None
        records.append(rec)

    df = pd.DataFrame(records)
    int_cols = [c for c in df.columns if c.startswith('Intermediate')]
    df = df.dropna(subset=int_cols, how='all').where(pd.notnull(df), None)
    return df.sort_values(['Starting Activity', 'Ending Activity'] + int_cols)

def get_xor_candidates(csv_path: str, support_threshold=0.01, max_path_length=3, activity_threshold=0.00001) -> pd.DataFrame:
    log = get_significant_activities(expand_prefix_csv_to_log(csv_path), activity_threshold)
    xor_candidates = defaultdict(lambda: {"count": defaultdict(int), "total": 0})

    for _, trace in log.groupby('case:concept:name'):
        events = trace.sort_values("position")["concept:name"].tolist()
        seen_triples, seen_pairs = set(), set()
        for i in range(len(events) - max_path_length + 1):
            pair, triple = (events[i], events[i + max_path_length - 1]), (events[i], events[i + 1], events[i + max_path_length - 1])
            if pair not in seen_pairs:
                xor_candidates[pair]["total"] += 1
                seen_pairs.add(pair)
            if triple not in seen_triples:
                xor_candidates[pair]["count"][events[i + 1]] += 1
                seen_triples.add(triple)

    records = []
    total_cases = log['case:concept:name'].nunique()
    for (start, end), data in xor_candidates.items():
        if len(data["count"]) > 1 and (data["total"] / total_cases) >= support_threshold:
            rec = {'Start Activity': start, 'End Activity': end, 'Num Alternatives': len(data["count"])}
            for i, alt in enumerate(sorted(data["count"].keys())):
                rec[f'Alternative {i + 1}'] = alt
            records.append(rec)
            
    df = pd.DataFrame(records)
    return df.where(pd.notnull(df), None)

def map_patterns_to_tokens(patterns_df: pd.DataFrame, x_word_dict: dict) -> pd.DataFrame:
    df = patterns_df.copy()
    cols = ['Starting Activity', 'Ending Activity'] + [c for c in df.columns if c.startswith('Intermediate')]
    for c in cols:
        df[c] = df[c].apply(lambda x: np.float32(x_word_dict[x]) if pd.notna(x) and x in x_word_dict else np.nan)
    return df

def map_xor_candidates_to_tokens(xor_df: pd.DataFrame, x_word_dict: dict) -> pd.DataFrame:
    df = xor_df.copy()
    cols = ['Start Activity', 'End Activity'] + [c for c in df.columns if c.startswith('Alternative')]
    for c in cols:
        df[c] = df[c].apply(lambda x: np.float32(x_word_dict[x]) if pd.notna(x) and x in x_word_dict else np.nan)
    return df

# ==============================================================================
# AUGMENTORS (From easy_augmentors.py and augmentors.py)
# ==============================================================================
class BaseAugmentor(abc.ABC):
    def augment(self, sequence: list) -> list:
        raise NotImplementedError()
    def is_applicable(self, sequence: list) -> bool:
        return len([x for x in sequence if x != 0]) > 2
    @staticmethod
    def get_name() -> str:
        raise NotImplementedError()

class RandomInsertion(BaseAugmentor):
    def __init__(self, available_tokens: list):
        self.available_tokens = available_tokens

    def is_applicable(self, sequence: list) -> bool:
        return len([x for x in sequence if x != 0]) >= 2 and len(self.available_tokens) > 0

    def augment(self, sequence: list) -> list:
        clean_seq = [x for x in sequence if x != 0]
        position = random.randint(1, len(clean_seq) - 1)
        augmented = deepcopy(clean_seq)
        augmented.insert(position, random.choice(self.available_tokens))
        return [0.0] * (len(sequence) - len(augmented)) + augmented

    @staticmethod
    def get_name() -> str: return "RandomInsertion"

class RandomDeletion(BaseAugmentor):
    def augment(self, sequence: list) -> list:
        clean_seq = [x for x in sequence if x != 0]
        position = random.randint(1, len(clean_seq) - 2)
        augmented = deepcopy(clean_seq)
        del augmented[position]
        return [0.0] * (len(sequence) - len(augmented)) + augmented

    @staticmethod
    def get_name() -> str: return "RandomDeletion"

class RandomReplacement(BaseAugmentor):
    def __init__(self, available_tokens: list):
        self.available_tokens = available_tokens

    def augment(self, sequence: list) -> list:
        clean_seq = [x for x in sequence if x != 0]
        position = random.randint(1, len(clean_seq) - 2)
        replacement = random.choice(self.available_tokens)
        while replacement == clean_seq[position]:
            replacement = random.choice(self.available_tokens)
        augmented = deepcopy(clean_seq)
        augmented[position] = replacement
        return [0.0] * (len(sequence) - len(augmented)) + augmented

    @staticmethod
    def get_name() -> str: return "RandomReplacement"

class RandomSwap(BaseAugmentor):
    def augment(self, sequence: list) -> list:
        clean_seq = [x for x in sequence if x != 0]
        pos1, pos2 = random.sample(range(1, len(clean_seq) - 1), 2)
        augmented = deepcopy(clean_seq)
        augmented[pos1], augmented[pos2] = augmented[pos2], augmented[pos1]
        return [0.0] * (len(sequence) - len(augmented)) + augmented

    @staticmethod
    def get_name() -> str: return "RandomSwap"

class StatisticalReplacement(BaseAugmentor):
    def __init__(self, patterns_df: pd.DataFrame):
        self.patterns_df = patterns_df
        self.alternative_cols = [col for col in patterns_df.columns if col.startswith("Alternative")]

    def is_applicable(self, sequence: list) -> bool:
        for _, row in self.patterns_df.iterrows():
            start, end = row['Start Activity'], row['End Activity']
            alternatives = [row[col] for col in self.alternative_cols if pd.notna(row[col])]
            for i in range(len(sequence) - 2):
                if sequence[i] == start and sequence[i + 2] == end:
                    mid = sequence[i + 1]
                    if mid in alternatives and any(a != mid for a in alternatives):
                        return True
        return False

    def augment(self, sequence: list) -> list:
        valid_positions = []
        for _, row in self.patterns_df.iterrows():
            start, end = row['Start Activity'], row['End Activity']
            alternatives = [row[col] for col in self.alternative_cols if pd.notna(row[col])]
            for i in range(len(sequence) - 2):
                if sequence[i] == start and sequence[i + 2] == end:
                    mid = sequence[i + 1]
                    if mid in alternatives:
                        alt_choices = [a for a in alternatives if a != mid]
                        if alt_choices:
                            valid_positions.append((i + 1, alt_choices))
                            
        pos, candidates = random.choice(valid_positions)
        augmented = deepcopy(sequence)
        augmented[pos] = random.choice(candidates)
        return augmented

    @staticmethod
    def get_name() -> str: return 'StatisticalReplacement'

class StatisticalDeletion(BaseAugmentor):
    def __init__(self, patterns_df: pd.DataFrame):
        self.patterns_df = patterns_df
        self.intermediate_cols = [col for col in patterns_df.columns if col.startswith("Intermediate")]

    def is_applicable(self, sequence: list) -> bool:
        for _, row in self.patterns_df.iterrows():
            start, end = row["Starting Activity"], row["Ending Activity"]
            intermediates = [int(row[col]) for col in self.intermediate_cols if pd.notna(row[col])]
            for i in range(len(sequence) - len(intermediates) - 1):
                window = sequence[i : i + len(intermediates) + 2]
                if window[0] == start and window[-1] == end and window[1:-1] == intermediates:
                    return True
        return False

    def augment(self, sequence: list) -> list:
        valid_matches = []
        for _, row in self.patterns_df.iterrows():
            start, end = row["Starting Activity"], row["Ending Activity"]
            intermediates = [int(row[col]) for col in self.intermediate_cols if pd.notna(row[col])]
            for i in range(len(sequence) - len(intermediates) - 1):
                window = sequence[i : i + len(intermediates) + 2]
                if window[0] == start and window[-1] == end and window[1:-1] == intermediates:
                    valid_matches.append((i + 1, i + 1 + len(intermediates)))
                    
        start_del, end_del = random.choice(valid_matches)
        return sequence[:start_del] + sequence[end_del:]

    @staticmethod
    def get_name() -> str: return 'StatisticalDeletion'

class StatisticalInsertion(BaseAugmentor):
    def __init__(self, patterns_df: pd.DataFrame):
        self.patterns_df = patterns_df
        self.intermediate_cols = [col for col in patterns_df.columns if col.startswith("Intermediate")]

    def is_applicable(self, sequence: list) -> bool:
        for _, row in self.patterns_df.iterrows():
            start, end = row['Starting Activity'], row['Ending Activity']
            for i in range(len(sequence) - 1):
                if sequence[i] == start and sequence[i + 1] == end:
                    return True
        return False

    def augment(self, sequence: list) -> list:
        valid_insertions = []
        for _, row in self.patterns_df.iterrows():
            start, end = row['Starting Activity'], row['Ending Activity']
            intermediates = [int(row[col]) for col in self.intermediate_cols if pd.notna(row[col])]
            for i in range(len(sequence) - 1):
                if sequence[i] == start and sequence[i + 1] == end:
                    valid_insertions.append((i, intermediates))
                    
        insert_pos, to_insert = random.choice(valid_insertions)
        return sequence[:insert_pos + 1] + to_insert + sequence[insert_pos + 1:]

    @staticmethod
    def get_name() -> str: return 'StatisticalInsertion'

# ==============================================================================
# AUGMENTATION LOGIC (From augmentation.py)
# ==============================================================================
def augment_training_set(sequences, main_augmentors, x_word_dict):
    available_activities = list(x_word_dict.values())
    fallback_augmentors = {
        "UsefulInsertion": RandomInsertion(available_activities),
        "UsefulDeletion": RandomDeletion(),
        "UsefulReplacement": RandomReplacement(available_activities),
        "ParallelSwap": RandomSwap()
    }

    original_list, x1_list, x2_list = [], [], []
    augmentor_usage_count = {aug.get_name(): 0 for aug in main_augmentors}
    for fallback in fallback_augmentors.values():
        augmentor_usage_count[fallback.get_name()] = 0
    augmentor_usage_count['No Augmentation'] = 0

    for sequence in sequences:
        random.shuffle(main_augmentors)
        x1_augmentor = random.choice(main_augmentors)
        x2_augmentor = random.choice([aug for aug in main_augmentors if aug != x1_augmentor])
        
        def try_augment(seq, augmentor):
            try:
                if augmentor.is_applicable(seq):
                    augmented_seq = augmentor.augment(seq)
                    augmentor_usage_count[augmentor.get_name()] += 1
                    return augmented_seq
                raise Exception("Not applicable")
            except Exception:
                fallback = fallback_augmentors.get(augmentor.get_name())
                if fallback and fallback.is_applicable(seq):
                    augmented_seq = fallback.augment(seq)
                    augmentor_usage_count[fallback.get_name()] += 1
                    return augmented_seq
            augmentor_usage_count['No Augmentation'] += 1
            return seq
        
        original_list.append(sequence.copy())
        x1_list.append(try_augment(sequence, x1_augmentor))
        x2_list.append(try_augment(sequence, x2_augmentor))

    print("\nAugmentor usage count:")
    for aug, count in augmentor_usage_count.items():
        print(f"{aug}: {count}")

    return original_list, x1_list, x2_list

# ==============================================================================
# PIPELINE INTEGRATION (From augmentation_strategy.py)
# ==============================================================================
class EdaAugmentation(abc.ABC):
    def __init__(self, activities: typing.Set[str], resources: typing.Set[str], event_log: pd.DataFrame, 
                 augmentors: typing.List[BaseAugmentor], augmentation_factor: float, allow_multiple: bool, 
                 record_augmentation: bool, dry_run: bool):
        self._activities, self._resources, self._event_log = activities, resources, event_log
        self._augmentors = augmentors
        self._augmentation_factor = augmentation_factor
        self._allow_multiple = allow_multiple
        self._record_augmentation = record_augmentation
        self._dry_run = dry_run

    @staticmethod
    def _get_trace_by_case_id(case_id: str, event_log: EventLog) -> typing.Union[Trace, None]:
        for trace in event_log:
            if trace.attributes['concept:name'] == case_id:
                return trace
        return None

    def augment(self) -> typing.Tuple[pd.DataFrame, typing.Dict, typing.Dict]:
        augmentation_count = {k.get_name(): 0 for k in self._augmentors}
        augmentation_record = {k.get_name(): [] for k in self._augmentors}

        if self._dry_run:
            return self._event_log, augmentation_count, augmentation_record

        augmented_event_log = pm4py.convert_to_event_log(self._event_log.__deepcopy__())
        case_ids = list(get_case_ids(self._event_log))
        traces_to_generate = math.ceil(self._augmentation_factor * len(case_ids)) - len(case_ids)

        i = 0
        while traces_to_generate > 0:
            random.shuffle(case_ids)
            trace = self._get_trace_by_case_id(case_ids[0], augmented_event_log if self._allow_multiple else pm4py.convert_to_event_log(self._event_log))
            random.shuffle(self._augmentors)

            if self._augmentors[0].is_applicable('', trace):
                try:
                    new_id = f'{case_ids[0]}_{i}'
                    augmented_trace = self._augmentors[0].augment(trace)
                    augmented_event_log.append(Trace(augmented_trace[:], attributes={'concept:name': new_id, 'creator': self._augmentors[0].get_name()}))
                    
                    if self._record_augmentation:
                        augmentation_count[self._augmentors[0].get_name()] += 1
                        augmentation_record[self._augmentors[0].get_name()].append(case_ids[0])
                    if self._allow_multiple:
                        case_ids.append(new_id)

                    traces_to_generate -= 1
                    i += 1
                except AssertionError:
                    continue

        if not self._allow_multiple:
            for trace in self._event_log:
                augmented_event_log.append(trace)

        return pm4py.convert_to_dataframe(augmented_event_log.__deepcopy__()), augmentation_count, augmentation_record

@dataclasses.dataclass
class EdaAugmentationConfig:
    augmentation: typing.Type[EdaAugmentation]
    augmentors: typing.List[BaseAugmentor]
    augmentation_factor: float
    allow_multiple: bool
    record_augmentation: bool

class EdaAugmenter(BasePipelineStep):
    def __init__(self, augmentations: typing.List[EdaAugmentationConfig], target_dir: str) -> None:
        super().__init__()
        self._augmentations = augmentations
        self._target_dir = target_dir

    def store_records(self, records: typing.List[dict]) -> None:
        with open(os.path.join(self._target_dir, 'record.json'), 'w', encoding='utf8') as f:
            json.dump(records, f, indent=4)

    def store_counts(self, counts: typing.List[dict]) -> None:
        with open(os.path.join(self._target_dir, 'count.json'), 'w', encoding='utf8') as f:
            json.dump(counts, f, indent=4)

    def run(self, train_aug: pd.DataFrame, context: AbstractPipelineContext, dry_run: bool = False) -> typing.List[pd.DataFrame]:
        all_activities = get_activities(context.get_full_event_log())
        all_resources = get_resources(context.get_full_event_log())

        all_augmentations, all_counts, all_records = [], [], []
        for aug in self._augmentations:
            augmenter = aug.augmentation(activities=all_activities, resources=all_resources, event_log=train_aug,
                                         augmentors=aug.augmentors, augmentation_factor=aug.augmentation_factor, 
                                         allow_multiple=aug.allow_multiple, record_augmentation=aug.record_augmentation, dry_run=dry_run)
            augmented_event_log, augmentation_count, augmentation_record = augmenter.augment()
            all_augmentations.append(augmented_event_log)
            all_counts.append(augmentation_count)
            all_records.append(augmentation_record)

        self.store_counts(all_counts)
        self.store_records(all_records)
        return all_augmentations

    @staticmethod
    def get_name() -> str:
        return 'Easy Data Augmentation Augmenter'