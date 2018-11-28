import json

vector_sizes = ['25', '50', '75', '100', '125', '150', '200']
metrics = ['mean_train_AUC', 'mean_train_Recall', 'mean_train_Accuracy', 'mean_train_Precision', 'mean_test_AUC',
           'mean_test_Recall', 'mean_test_Accuracy', 'mean_test_Precision']
all_vectors_means = {}


def start():
    """
    Initialize dictionary with all vector sizes. Each vector size has its own, over 10-nested folds averaged metrics.
    """
    metrics_dict = {
        "mean_train_AUC": 0.0,
        "mean_train_Recall": 0.0,
        "mean_train_Accuracy": 0.0,
        "mean_train_Precision": 0.0,
        "mean_test_AUC": 0.0,
        "mean_test_Recall": 0.0,
        "mean_test_Accuracy": 0.0,
        "mean_test_Precision": 0.0
    }
    for size in range(vector_sizes.__len__()):
        all_vectors_means.setdefault(vector_sizes[size], metrics_dict)


def normalize():
    """
    Normalize sum for k-folds
    """
    # Later when models available
    # for i in ['25', '50', '75', '100', '125', '150', '200']: or in vector_sizes.len
    for i in ['25']:
            for m in metrics:
                all_vectors_means[i][m.__str__()] = all_vectors_means[i][m.__str__()] / 10


start()
# Later when models available
#for i in ['25', '50', '75', '100', '125', '150', '200']:
for i in ['25']:
    for j in range(9):
        for metric in metrics:
            data = json.loads(open('./' + i + '/cv_result_' + i + '_outer_split_' + j.__str__() + '.json').read())
            value = data[metric.__str__()]
            # Add up the values for a specific parameter in all folds
            all_vectors_means[i][metric.__str__()] = value['__ndarray__'][0] + all_vectors_means[i][metric.__str__()]

normalize()
print(all_vectors_means)


