import os
import json
import pprint
import pandas as pd
import numpy as np
import argparse
import re


def entry_to_np_array(entry):
    if "__ndarray__" not in entry or "dtype" not in entry:
        print("entry not of right shape:", entry)
        return
    return np.array(entry["__ndarray__"], dtype=entry["dtype"])


def extract_data(json_file, outer_fold=0, vector_size=0):

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

parser = argparse.ArgumentParser(description='Clean JSON data dump of scikit-learn output')
parser.add_argument('-d', '--data_dir')
parser.add_argument('-o', '--results_tsv')
args = parser.parse_args()


#print(os.getcwd())
all_df = []
# Vector sizes
#for vsize in ['25', '50', '100', '125', '200']:
#    for fold in range(10):
#        json_file = '../../results/{}/cv_result_{}_outer_split_{}.json'.format(vsize, vsize, fold)
#        print(json_file)
#        all_df.append(extract_data(json_file, outer_fold=fold, vector_size=vsize))

# Different number of layers
for json_file in os.listdir(args.data_dir):
    if not json_file.endswith(".json"):
        pass
    json_file = args.data_dir + "/" + json_file
    print("file:", json_file)
    outer_fold = re.match(r".* ?_(.*)_.*", json_file).group(1)
    layers = re.match(r".* ?\((.*)\).*", json_file).group(1)
    all_df.append(extract_data(json_file, outer_fold=outer_fold, vector_size=layers))


complete_df = pd.concat(all_df)
print(complete_df)
complete_df.to_csv(args.results_tsv, sep='\t')
