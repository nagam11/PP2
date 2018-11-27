import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import calibration_curve
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from sklearn.metrics import make_scorer, accuracy_score, recall_score, precision_score


def norm(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))


def resample(sample_idx, data_y, oversample=True):
    pos_idx = np.array([idx for idx in sample_idx if data_y[idx] == 1])
    neg_idx = np.array([idx for idx in sample_idx if data_y[idx] == 0])

    num_pos = pos_idx.size
    num_neg = neg_idx.size

    rnd = np.random.RandomState(42)

    if num_pos < num_neg:
        if oversample:
            n = num_neg // num_pos
            d = num_neg % num_pos

            new_idx = rnd.choice(pos_idx, size=d, replace=False)
            new_idx = np.concatenate((new_idx, np.repeat(pos_idx, n), neg_idx))
        else:
            new_idx = rnd.choice(neg_idx, size=num_pos, replace=False)
            new_idx = np.concatenate((new_idx, pos_idx))
    elif num_pos > num_neg:
        if oversample:
            n = num_pos // num_neg
            d = num_pos % num_neg

            new_idx = rnd.choice(neg_idx, size=d, replace=False)
            new_idx = np.concatenate((new_idx, np.repeat(neg_idx, n), pos_idx))
        else:
            new_idx = rnd.choice(pos_idx, size=num_neg, replace=False)
            new_idx = np.concatenate((new_idx, neg_idx))
    else:
        return sample_idx

    rnd.shuffle(new_idx)

    return new_idx


def parse_cv_results(cv_results, params_scores):
    for i, params in enumerate(cv_results['params']):
        if repr(params) not in params_scores:
            params_scores[repr(params)] = []

        fold_idx = 0

        while True:
            key = 'split{}_test_score'.format(fold_idx)
            scores = cv_results.get(key, None)

            if scores is None:
                break
            else:
                fold_idx += 1

            params_scores[repr(params)].append(scores[i])


def train_and_optimize(data_x, data_y):
    rs1 = RobustScaler()
    rs2 = RobustScaler()
    pca = PCA(n_components=10,
              svd_solver='full',
              whiten=False,
              random_state=42)
    '''
    clf = SVC(kernel='rbf',
              gamma='auto',
              cache_size=1000,
              probability=False,
              class_weight='balanced',
              decision_function_shape='ovr',
              random_state=42)
    '''
    clf = MLPClassifier(solver='adam',
                        activation='relu',
                        hidden_layer_sizes=(50, ),
                        beta_1=0.9,
                        beta_2=0.999,
                        max_iter=200,
                        early_stopping=True,
                        validation_fraction=0.2,
                        random_state=42)
    model = Pipeline(steps=[('rs1', rs1), ('pca', pca),
                            ('rs2', rs2), ('clf', clf)])
    kfold1 = StratifiedKFold(n_splits=10, random_state=42)
    kfold2 = StratifiedKFold(n_splits=3, random_state=42)
    params = {# 'clf__C': [1e-8, 1e-6, 1e-4, 1e-2, 1, 10],
              'clf__alpha': [0.1, 1e-2, 1e-3, 1e-4],
              'clf__epsilon': [1, 1e-4, 1e-8],
              'clf__learning_rate_init': [0.1, 1e-2, 1e-3, 1e-4],
              }

    cv_test = []
    cv_train = []
    cv_cross = []
    params_scores = {}

    all_cv_results = ''
    ref_results = ''

    scoring = {'Accuracy': make_scorer(accuracy_score),
               'Precision': make_scorer(precision_score),
               'Recall': make_scorer(recall_score),
               'AUC': make_scorer(roc_auc_score)}

    for i, (train, test) in enumerate(kfold1.split(data_x, data_y)):
        print('CV-Split {}'.format(i + 1))

        splits = []

        for x1, x2 in kfold2.split(data_x[train], data_y[train]):
            splits.append((resample(x1, data_y[train], oversample=False), x2))

        gridcv = GridSearchCV(model, params, cv=splits, n_jobs=3, refit=True,
                              scoring=scoring)

        gridcv.fit(data_x[train], data_y[train])

        print(gridcv.best_estimator_.named_steps['clf'].n_iter_)

        s_test = gridcv.score(data_x[test], data_y[test])
        s_train = gridcv.score(data_x[train], data_y[train])

        results = 'kfold1_split: ' + str(i) + '\n'
        results += '###' + '\n'
        results += 'Best Params: {}'.format(gridcv.best_params_) + '\n'
        results += 'Train Score: {}'.format(s_train) + '\n'
        results += 'Cross Score: {}'.format(gridcv.best_score_) + '\n'
        results += 'Test Score: {}'.format(s_test) + '\n'
        results += '###' + '\n'
        ref_results += results
        print(results)

        cv_test.append(s_test)
        cv_train.append(s_train)
        cv_cross.append(gridcv.best_score_)

        all_cv_results = gridcv.cv_results_
        parse_cv_results(gridcv.cv_results_, params_scores)

    test_mean, test_std = np.mean(cv_test), np.std(cv_test, ddof=1)
    train_mean, train_std = np.mean(cv_train), np.std(cv_train, ddof=1)
    cross_mean, cross_std = np.mean(cv_cross), np.std(cv_cross, ddof=1)

    results = '###' + '\n'
    results += 'CV Train Score: {} ({})'.format(train_mean, train_std) + '\n'
    results += 'CV Cross Score: {} ({})'.format(cross_mean, cross_std) + '\n'
    results += 'CV Test Score: {} ({})'.format(test_mean, test_std) + '\n'
    results += '###' + '\n'
    ref_results += results
    print(results)

    params_stats = [(np.mean(s), np.std(s, ddof=1), p)
                    for p, s in params_scores.items()]
    params_stats = sorted(params_stats, reverse=True)

    final_mean, final_std, final_params = params_stats[0]

    results = '###' + '\n'
    results += 'Final Params: {}'.format(final_params) + '\n'
    results += 'Final Cross Score: {} ({})'.format(final_mean, final_std) + '\n'
    results += '###' + '\n'
    ref_results += results
    print(results)

    return all_cv_results, ref_results


def cross_validation(data_x, data_y):
    rs1 = RobustScaler()
    rs2 = RobustScaler()
    pca = PCA(n_components=10,
              svd_solver='full',
              whiten=False,
              random_state=42)
    '''
    clf = SVC(kernel='rbf',
              C=1e-8,
              gamma='auto',
              cache_size=1000,
              probability=False,
              class_weight='balanced',
              decision_function_shape='ovr',
              random_state=42)
    '''
    clf = MLPClassifier(solver='adam',
                        activation='relu',
                        hidden_layer_sizes=(50, ),
                        alpha=0.1,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-4,
                        learning_rate_init=1e-3,
                        max_iter=200,
                        early_stopping=True,
                        validation_fraction=0.2,
                        random_state=42)

    kfold = StratifiedKFold(n_splits=10, random_state=42)

    cv_test_auc = []
    cv_train_auc = []
    cv_test_brier = []
    cv_train_brier = []

    fig = plt.figure(figsize=(10, 10))

    plt.ylim([-0.05, 1.05])
    plt.plot([0, 1], [0, 1], 'k:', label='Perfect')

    for i, (train, test) in enumerate(kfold.split(data_x, data_y)):
        print('CV-Split {}'.format(i + 1))

        train = resample(train, data_y, oversample=False)

        model = Pipeline(steps=[('rs1', rs1), ('pca', pca),
                                ('rs2', rs2), ('clf', clf)])

        model.fit(data_x[train], data_y[train])

        '''
        Get all your statistics here.
        For example: AUC, Brier loss and the calibration curve.
        '''
        if hasattr(model, 'predict_proba'):
            p_test = model.predict_proba(data_x[test])[:, 1]
            p_train = model.predict_proba(data_x[train])[:, 1]
        else:
            p_test = model.decision_function(data_x[test])
            p_train = model.decision_function(data_x[train])
            p_test = norm(p_test)
            p_train = norm(p_train)

        p_pos, s_mean = calibration_curve(data_y[test], p_test, n_bins=10)

        plt.plot(s_mean, p_pos, 's-', label='CV fold {}'.format(i + 1))

        test_auc = roc_auc_score(data_y[test], p_test)
        train_auc = roc_auc_score(data_y[train], p_train)

        test_brier = brier_score_loss(data_y[test], p_test)
        train_brier = brier_score_loss(data_y[train], p_train)

        print('###')
        print('Train AUC: {}'.format(train_auc))
        print('Test AUC: {}'.format(test_auc))
        print('Train Brier Loss: {}'.format(train_brier))
        print('Test Brier Loss: {}'.format(test_brier))
        print('###')

        cv_test_auc.append(test_auc)
        cv_train_auc.append(train_auc)
        cv_test_brier.append(test_brier)
        cv_train_brier.append(train_brier)

    plt.title('Calibration plot  (reliability curve)')
    plt.ylabel('Fraction of positives')
    plt.xlabel('Mean predicted value')
    plt.legend(loc='best', ncol=2)
    plt.savefig('calibration.png')
    plt.close(fig)

    test_auc_stats = np.mean(cv_test_auc), np.std(cv_test_auc, ddof=1)
    train_auc_stats = np.mean(cv_train_auc), np.std(cv_train_auc, ddof=1)
    test_brier_stats = np.mean(cv_test_brier), np.std(cv_test_brier, ddof=1)
    train_brier_stats = np.mean(cv_train_brier), np.std(cv_train_brier, ddof=1)

    print('###')
    print('CV Train AUC: {0[0]} ({0[1]})'.format(train_auc_stats))
    print('CV Test AUC: {0[0]} ({0[1]})'.format(test_auc_stats))
    print('CV Train Brier Loss: {0[0]} ({0[1]})'.format(train_brier_stats))
    print('CV Test Brier Loss: {0[0]} ({0[1]})'.format(test_brier_stats))
    print('###')
