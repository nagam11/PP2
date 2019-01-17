import copy
import json
import matplotlib.pyplot as plt
import numpy as np

vector_sizes = ["25", "50", "75", "100", "125", "150", "200"]
metrics = ['mean_train_AUC', 'mean_train_Recall', 'mean_train_Accuracy', 'mean_train_Precision', 'mean_test_AUC',
           'mean_test_Recall', 'mean_test_Accuracy', 'mean_test_Precision']
all_vectors_means = {}


def start():
    """
    Initialize dictionary with all vector sizes. Each vector size has its own, over 10-nested folds averaged metrics.
    """
    metrics_dict = {
        "mean_train_AUC": [],
        "mean_train_Recall": [],
        "mean_train_Accuracy": [],
        "mean_train_Precision": [],
        "mean_test_AUC": [],
        "mean_test_Recall": [],
        "mean_test_Accuracy": [],
        "mean_test_Precision": [],
        "max_index": None
    }
    for size in range(vector_sizes.__len__()):
        all_vectors_means.setdefault(vector_sizes[size], copy.deepcopy(metrics_dict))


def normalize():
    """
    Normalize sum for k-folds
    """
    # Later when models available
    # for i in vector_sizes
    for i in ['50', '100', '200']:
        for m in metrics:
            all_vectors_means[i][m.__str__()] = all_vectors_means[i][m.__str__()] / 10


start()
# Later when models available
for i in vector_sizes:
    for j in range(10):
        for metric in metrics:
            data = json.loads(open('../../results/' + i + '/cv_result_' + i + '_outer_split_' + j.__str__() + '.json').read())
            value = data[metric.__str__()]
            val = value['__ndarray__']
            all_vectors_means[i][metric.__str__()].append(val)

max_dict = {}
index = 0
for key, metrics in all_vectors_means.items():
    for metric, array in metrics.items():
        if metric not in "max_index" and len(array) != 0:
            all_vectors_means[key][metric] = np.concatenate(array)

        if "mean_test_AUC" in metric and len(array) != 0:
            all_vectors_means[key].update({"max_index": np.argmax(all_vectors_means[key][metric])})
    for metric, array in metrics.items():
        if metric not in "max_index" and len(array) != 0:
            all_vectors_means[key][metric] = all_vectors_means[key][metric][index]

print(index)
#normalize()

metrics = ["AUC", "Recall", "Precision", "Accuracy"]
for m in metrics:
    train_means = []
    test_means = []
    for key in vector_sizes:
        s = 'mean_train_' + m
        s1 = 'mean_test_' + m
        val = round(all_vectors_means[key][s], 4)
        val1 = round(all_vectors_means[key][s1], 4)
        train_means.append(val)
        test_means.append(val1)

    train_means = tuple(train_means)
    test_means = tuple(test_means)

    ind = np.arange(len(train_means))  # the x locations for the groups
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(ind - width / 2, train_means, width,
                    color='SkyBlue', label='Train')
    rects2 = ax.bar(ind + width / 2, test_means, width,
                    color='IndianRed', label='Test')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Best AUC')
    ax.set_title('AUC by vector size for train and test')
    ax.set_xticks(ind)
    plt.ylim((0, 0.7))
    ax.set_xticklabels(('25', '50', '75', '100', '125', '150' '200', 'G4', 'G5'))
    ax.legend()


    def autolabel(rects, xpos='center'):
        """
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.57, 'left': 0.43}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                    '{}'.format(height), ha=ha[xpos], va='bottom')


    autolabel(rects1, "left")
    autolabel(rects2, "right")

    plt.show()