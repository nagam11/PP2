import os
import json
import pprint
import pandas as pd
import numpy as np


def entry_to_np_array(entry):
    if "__ndarray__" not in entry or "dtype" not in entry:
        print("entry not of right shape:", entry)
        return
    return np.array(entry["__ndarray__"], dtype=entry["dtype"])


def extract_data(json_file, outer_fold, vector_size):

    with open(json_file) as json_data:
        d = json.load(json_data)

    mean_results_train = {}
    mean_results_test = {}

    for key, value in d.items():
        if key == "params":
            params = pd.DataFrame.from_dict(value)

        elif "mean" in key:
            if "train" in key:
                if "AUC" in key:
                    mean_results_train["AUC"] = entry_to_np_array(value)
                elif "Accuracy" in key:
                    mean_results_train["Accuracy"] = entry_to_np_array(value)
                elif "Recall" in key:
                    mean_results_train["Recall"] = entry_to_np_array(value)
                elif "Precision" in key:
                    mean_results_train["Precision"] = entry_to_np_array(value)
            elif "test" in key:
                if "AUC" in key:
                    mean_results_test["AUC"] = entry_to_np_array(value)
                elif "Accuracy" in key:
                    mean_results_test["Accuracy"] = entry_to_np_array(value)
                elif "Recall" in key:
                    mean_results_test["Recall"] = entry_to_np_array(value)
                elif "Precision" in key:
                    mean_results_test["Precision"] = entry_to_np_array(value)

    train_df = pd.DataFrame.from_dict(mean_results_train)
    test_df = pd.DataFrame.from_dict(mean_results_test)

    test_df["validation"] = "test"
    train_df["validation"] = "train"
    total_df = pd.concat([test_df, train_df])
    total_df["outer_split"] = outer_fold
    total_df["vector_size"] = vector_size

    return total_df


print(os.getcwd())
all_df = []
for vsize in ['25', '50', '100', '125', '200']:
    for fold in range(10):
        json_file = '../../results/{}/cv_result_{}_outer_split_{}.json'.format(vsize, vsize, fold)
        print(json_file)
        all_df.append(extract_data(json_file, outer_fold=fold, vector_size=vsize))

complete_df = pd.concat(all_df)
complete_df.to_csv("../../results/mean_stats_data_table.tsv", sep='\t')
