import json

# Metrics needed for extraction
metrics = ['mean_train_AUC', 'mean_train_Recall', 'mean_train_Accuracy', 'mean_train_Precision', 'mean_test_AUC',
           'mean_test_Recall', 'mean_test_Accuracy', 'mean_test_Precision']

# Later when models available
#for i in ['25', '50', '75', '100', '125', '150', '200']:
for i in ['25']:
    for j in range(9):
        for m in metrics:
            data = json.loads(open('./' + i + '/cv_result_' + i + '_outer_split_' + j.__str__() + '.json').read())
            value = data[m.__str__()]
            print(value['__ndarray__'][0])
