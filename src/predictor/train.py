import os

import csv

import random

import time

import joblib

import numpy as np

from scipy.stats import randint, uniform, loguniform

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, fbeta_score, make_scorer

from sklearn.feature_selection import VarianceThreshold, RFECV

from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import RandomizedSearchCV, KFold

from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE, ADASYN

from imblearn.under_sampling import RandomUnderSampler

from sklearn.base import clone

from lightgbm import LGBMClassifier

def get_dataset_from_csv_file(data_file: str):
    """
    Get dataset from csv file
    Format of csv file(with headrow): id, delta_n, dij, feat1, feat2, feat3 ...
    Args:
        data_file(str): path to csv file
    Returns:
        a tuple of (feature_names(list), ids(list), dns(ndarray), dijs(ndarray), features(ndarray))
        if dn/dij is empty: hold the place with 9999.999
    """
    assert os.path.exists(data_file), 'File does not exist: {}'.format(data_file)
    assert data_file.endswith('.csv'), 'File should be a csv file'
    dataset = []
    with open(data_file, 'r') as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            dataset.append(row)
    n_rows = len(dataset)
    feature_names = dataset[0][3:]
    ids, dns, dijs = [], [], []
    for i in range(1, n_rows):
        ids.append(str(dataset[i][0]))
        str_dn = dataset[i][1]
        try:
            dns.append(float(str_dn))
        except:
            dns.append(9999.999)
        str_dij = dataset[i][2]
        try:
            dijs.append(float(str_dij))
        except:
            dijs.append(9999.999)
    dns, dijs = np.array(dns), np.array(dijs)
    features = np.array([dataset[i][3:] for i in range(1, n_rows)], dtype='float64')
    return (feature_names, ids, dns, dijs, features)

def shuffle_set(to_shuffle_set, random_seed=26):
    """
    Shuffle set with numpy random seed
    Args:
        to_shuffle_set(indexable objects like list or ndarray): set of elements to be shuffled
        random_seed(int): random seed
    Returns:
        shuffled_set(ndarray): shuffled set of elements
    """
    np.random.seed(random_seed)
    indices = np.array([i for i in range(len(to_shuffle_set))])
    np.random.shuffle(indices)
    shuffle_dataset = []
    for i in indices:
        shuffle_dataset.append(to_shuffle_set[i])
    return np.array(shuffle_dataset)

def shuffle_sets(to_shuffle_sets: list, seeds=[8, 9, 10]):
    """
    Multiply shuffle multiple sets of the same length, with the number of shuffles equal to the number of seeds
    Args:
        to_shuffle_sets(list): list of sets to be shuffled; each set has the same length(n_samples)
        seeds(list of int, default=[8,9,10]): random seeds
    Returns:
        shuffled_sets(list of ndarray): multiple shuffled sets of elements
    """
    n_samples = len(to_shuffle_sets[0])
    for to_shuffle_set in to_shuffle_sets:
        assert len(to_shuffle_set) == n_samples
    for seed in seeds:
        shuffled_sets = []
        assert type(seed) is int, '\'seeds\' must be a list of integers'
        for set_idx in range(len(to_shuffle_sets)):
            to_shuffle_set = shuffle_set(to_shuffle_sets[set_idx], seed)
            shuffled_sets.append(to_shuffle_set)
        to_shuffle_sets = shuffled_sets
    return shuffled_sets

def bin_cls(y_reg, c1, c2=None):
    """
    Convert continuous labels to binary labels; positive class(1): c1<=y(<c2)
    Args:
        y_reg(indexable objects like list or ndarray): set of continuous labels
        c1(int or float): lower bound for the positive class
        c2(if not None, int or float): upper bound for the positive class
    Returns:
        y_cls(ndarray): binary labels
    """
    y_cls = []
    if c2 is not None:
        for i in y_reg:
            if c1 <= i < c2:
                y_cls.append(1)
            else:
                y_cls.append(0)
    if c2 is None:
        for i in y_reg:
            if c1 <= i:
                y_cls.append(1)
            else:
                y_cls.append(0)
    y_cls = np.array(y_cls)
    return y_cls

def bin_cls_evl(y_true, y_pred, beta=2, print_result=True):  # 二分类器评价
    """
    Performance metrics for binary classification task
    metrics: accuracy(A), precision(P), recall(R), roc_auc(AUC), fbeta_score(Fbeta)
    Args:
        y_true(list or ndarray): ground truth labels
        y_pred(list or ndarray): predicted labels
        beta(float, default=2.0): determines the weight of recall in the combined score
        print_result(bool, default=True): whether to print evaluation metrics with 4 decimal places
    Returns:
        list of metrics including accuracy, precision, recall, roc_auc, fbeta_score
    """
    A = accuracy_score(y_true, y_pred)
    P = precision_score(y_true, y_pred)
    R = recall_score(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_pred)
    Fbeta = fbeta_score(y_true, y_pred, beta=beta)
    if print_result:
        print('%.4f\t%.4f\t%.4f\t%.4f\t%.4f' % (A, P, R, AUC, Fbeta))
    return [A, P, R, AUC, Fbeta]

def get_sub_features_via_names(features, feature_names, selected_feature_names):
    """
    Get sub set of features from feature names
    Args:
        features(ndarray): original dataset of features(n_samples, len(feature_names))
        feature_names(list or ndarray): original feature names
        selected_feature_names(list or ndarray): feature names to keep
    Returns:
        selected_features(ndarray): subset of features(n_samples, len(selected_feature_names))
    """
    assert len(features[0]) == len(feature_names)
    feature_names = list(feature_names)
    selected_features = []
    for feature_name in selected_feature_names:
        feature_idx = feature_names.index(feature_name)
        selected_features.append(features[..., feature_idx])
    return np.array(selected_features).T

def vt_features(X_train, feature_names, threshold=0.03):
    """
    Feature selection removing normalized features with variance lower than threshold
    Args:
        X_train(ndarray): original dataset(should be training set) of features(n_samples, len(feature_names))
        feature_names(list or ndarray): original feature names
        threshold(float, default=0.3): threshold of variance that must be met by retained normalized features
    Returns:
        feat_name_vt(ndarray): names of features after normalization with variance higher than threshold
        print: number of selected/original features
    """
    X_train_norm = MinMaxScaler().fit_transform(X_train)
    vt = VarianceThreshold(threshold=threshold)
    vt.fit_transform(X_train_norm)
    feat_idx_vt = vt.get_support()
    feat_name_vt = []
    for i in range(len(feature_names)):
        if feat_idx_vt[i]:
            feat_name_vt.append(feature_names[i])
    feat_name_vt = np.array(feat_name_vt)
    print(f'Number of features after/before vt: {len(feat_name_vt)}/{len(feature_names)}')
    return feat_name_vt

def feat_ipt_cv_features(X_train, y_train, feature_names, n_features_to_keep=50, cv=5, estimator=None):
    """
    Feature selection by feature importance; retain features with higher importance
    The average of the importance of the features fitted by the estimator in each cross-validation
    is taken as the basis for ranking the importance of each feature.
    Args:
        X_train(ndarray): original dataset(should be training set) of features(n_samples, len(feature_names))
        y_train(list or ndarray): ground truth class labels
        feature_names(list or ndarray): original feature names
        n_features_to_keep(int, default=50): number of features to keep
        cv(int, default=5): number of cross-validation folds
        estimator(sklearn classifier, if None: GradientBoostingClassifier(random_state=42)): classifier that provides
            information of feature importance
    Returns:
        feat_name_ficv(ndarray): names of kept features with the highest importance
    """
    assert n_features_to_keep <= len(feature_names), f'n_features_to_keep={n_features_to_keep} is out of range'
    time0 = time.time()
    X, y = X_train, y_train
    kf = KFold(n_splits=cv, shuffle=False)
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    for train_idx, test_idx in kf.split(X, y):
        X_trains.append(X[train_idx])
        X_tests.append(X[test_idx])
        y_trains.append(y[train_idx])
        y_tests.append(y[test_idx])
    feature_importances_cv = []
    if estimator is None:
        estimator = GradientBoostingClassifier(random_state=42)
    for i in range(cv):
        X_train, X_test, y_train, y_test = X_trains[i], X_tests[i], y_trains[i], y_tests[i]
        estimator.fit(X_train, y_train)
        feature_importances_cv.append(estimator.feature_importances_)
    feature_importances_cv = np.array(feature_importances_cv)
    feature_importances = list(np.sum(feature_importances_cv, axis=0))
    assert len(feature_importances) == len(feature_names)
    sorted_indices = sorted(range(len(feature_importances)), key=lambda i: feature_importances[i], reverse=True)
    sorted_feature_names = [feature_names[i] for i in sorted_indices]
    feat_name_ficv = sorted_feature_names[:n_features_to_keep]
    time1 = time.time()
    print(f'Number of features after/before ficv: {len(feat_name_ficv)}/{len(feature_names)}; '
          f'time spent on ficv: {round((time1-time0), 3)}s')
    return np.array(feat_name_ficv)

def sets_split(ids, features, targets, train_ratio=0.8):
    """
    Split shuffled sets of ids, features and targets into training and test sets
    Samples with a placeholder value target(9999.999) will be removed before splitting
    Args:
        ids(1D ndarray): ids of all the crystal samples(n_samples)
        features(ndarray): features of all the crystal samples(n_samples, n_features)
        targets(1D ndarray): targets of all the crystal samples(n_samples)
        train_ratio(float, default=0.8): ratio of training set samples to total samples
    Returns:
        a tuple of ndarray(ids_train, X_train, y_train, ids_test, X_test, y_test)
    """
    assert len(features) == len(targets)
    chk_ids, chk_features, chk_targets = [], [], []
    for sample_idx in range(len(features)):
        if targets[sample_idx] != 9999.999:
            chk_ids.append(ids[sample_idx])
            chk_features.append(features[sample_idx])
            chk_targets.append(targets[sample_idx])
    chk_ids, chk_features, chk_targets = np.array(chk_ids), np.array(chk_features), np.array(chk_targets)
    n_train = round(len(chk_targets) * train_ratio)
    ids_train, X_train, y_train = chk_ids[:n_train], chk_features[:n_train], chk_targets[:n_train]
    ids_test, X_test, y_test = chk_ids[n_train:], chk_features[n_train:], chk_targets[n_train:]
    if len(features) != len(chk_features):
        print(f"{len(chk_features)} samples with valid targets from {len(features)}")
    return (ids_train, X_train, y_train, ids_test, X_test, y_test)

def rscv_hparams(X_train, y_train, estimator=None, hparams=None, cv=5, scorer=None):
    """
    Get estimator with the best hyperparameters using RandomSearchCV
    Args:
        X_train(ndarray): features of training set
        y_train(ndarray): class labels of training set
        estimator(sklearn classifier, if None: GradientBoostingClassifier(random_state=42)):
            classifier to optimize hyperparameters
        hparams(dict): hyperparameters range of estimator
        cv(int): number of cross-validation folds
        scorer(str or callable): scorer in RandomizedSearchCV; if None, adopt F2.0
    Returns:
        best_estimator(sklearn classifier): classifier with the best hyperparameters
        print the best hyperparameters
    """
    if estimator is None:
        estimator = GradientBoostingClassifier(random_state=42)
        # estimator = RandomForestClassifier(random_state=42)
    if hparams is None:
        hparams = {
            'n_estimators': [5*i for i in range(2, 10)],
            # 'learning_rate': [0.02*i for i in range(1, 9)],
            'max_depth': [i for i in range(3, 10)],
            'min_samples_split': [2, 3, 4, 5, 6],
            'min_samples_leaf': [1, 2, 3, 4, 5],
            # 'validation_fraction': [0.1, 0.2, 0.3, 0.4, 0.5],
            # 'tol': [1e-4, 1e-3, 1e-2, 1e-1],
            # 'ccp_alpha': [0.0],
            # 'subsample': [1.0],
            # 'min_impurity_decrease': [0.0],
            # 'max_features': ['auto', 'sqrt', 'log2', 2, 3, 4, 5],
        }
    if scorer is None:
        scorer = make_scorer(fbeta_score, beta=2)
    rscv = RandomizedSearchCV(estimator=estimator, param_distributions=hparams, cv=cv, random_state=26, scoring=scorer)
    time0 = time.time()
    rscv.fit(X_train, y_train)
    best_hparams = rscv.best_params_
    best_estimator = rscv.best_estimator_
    time1 = time.time()
    print(f'Best params: {best_hparams}; time spent on RSCV: {round((time1-time0), 3)}s')
    return best_estimator

def train_evaluate_and_save_model(estimator, X_train, y_train, X_test, y_test, imb_sample=True,
                                  beta=2, save_path=None, save_name=None):
    """
    Train the model, evaluate the performance and save the model as a pkl file
    Args:
        estimator(sklearn classifier): model with hyperparameters to train
        X_train(ndarray): features of training set
        y_train(ndarray): class labels of training set
        X_test(ndarray): features of test set
        y_test(ndarray): class labels of test set
        imb_sample(bool, default=True): whether to perform SMOTE oversampling when categories are imbalanced
                imbalance: more than 60% in any category
        beta(float, default=2.0): determines the weight of recall in the combined score
        save_path(str, default=None): path to save the model
        save_name(str, default=None): name to save the model
                if None, the model will be named as 'model.pkl' in save_path
    Returns:
        no returns; print the shape of X_train, evaluation metrics and SMOTE/saving message
    """
    print(f'Shape of X_train is {X_train.shape}')
    if imb_sample:
        if sum(y_train)/len(y_train) >= 0.6 or sum(y_train)/len(y_train) <= 0.4:
            smote = SMOTE(random_state=42)
            # smote = ADASYN(random_state=42)
            X_train_imb, y_train_imb = smote.fit_resample(X_train, y_train)
            print(f'SMOTE resampled X_train is {X_train_imb.shape}')
        else:
            X_train_imb, y_train_imb = X_train, y_train
        # elif sum(y_train)/len(y_train) < 0.4:
        #     undersample = RandomUnderSampler(random_state=42)
        #     X_train, y_train = undersample.fit_resample(X_train, y_train)
        #     print(f'RandUnder resampled X_train is {X_train.shape}')
    estimator.fit(X_train_imb, y_train_imb)
    print('A', '\t', 'P', '\t', 'R', '\t', 'AUC', '\t', f'F{beta}')
    bin_cls_evl(y_train, estimator.predict(X_train), beta)
    bin_cls_evl(y_test, estimator.predict(X_test), beta)
    if save_path is not None:
        assert os.path.exists(save_path)
        if save_name is not None:
            save_file = os.path.join(save_path, save_name + '.pkl')
        else:
            save_file = os.path.join(save_path, 'model.pkl')
        joblib.dump(estimator, save_file)
        print(f'Trained model is saved to {save_file}')

def nth_largest_idx(lst, nth=2):
    """
    Return the index of the n^rd largest value in a list/ndarray
    Args:
        lst(list/np.ndarray): list/ndarray to be sorted
        nth(int): an integer no larger than len(lst)
    Returns:
        The index of the n^rd largest value in a list/ndarray
    """
    lst = np.array(lst)
    assert len(lst) >= nth, f"nth={nth} should be no larger than lst={lst}"
    return np.argsort(lst)[-nth]

def train_cross_evaluate_and_save_model(estimator, X, y, imb_sample=True,
                                        cv=10, beta=2, save_path=None, save_name=None):
    """
    Train the model, evaluate the performance and save the model as a pkl file
    Args:
        estimator(sklearn classifier): model with hyperparameters to train
        X_train(ndarray): features of training set
        y_train(ndarray): class labels of training set
        X_test(ndarray): features of test set
        y_test(ndarray): class labels of test set
        imb_sample(bool, default=True): whether to perform SMOTE oversampling when categories are imbalanced
                imbalance: more than 60% in any category
        beta(float, default=2.0): determines the weight of recall in the combined score
        save_path(str, default=None): path to save the model
        save_name(str, default=None): name to save the model
                if None, the model will be named as 'model.pkl' in save_path
    Returns:
        no returns; print the shape of X_train, evaluation metrics and SMOTE/saving message
    """
    estimator = clone(estimator)
    print(f'Shape of X is {X.shape}')
    kf = KFold(n_splits=cv, shuffle=False)
    X_trains, X_tests, y_trains, y_tests = [], [], [], []
    for train_idx, test_idx in kf.split(X, y):
        X_trains.append(X[train_idx])
        X_tests.append(X[test_idx])
        y_trains.append(y[train_idx])
        y_tests.append(y[test_idx])
    metrics_train, metrics_test = [], []
    accuracy_train, precision_train, recall_train, auc_train, fbeta_train = [], [], [], [], []
    accuracy_test, precision_test, recall_test, auc_test, fbeta_test = [], [], [], [], []
    for i in range(cv):
        X_train, y_train, X_test, y_test = X_trains[i], y_trains[i], X_tests[i], y_tests[i]
        if imb_sample:
            if sum(y_train)/len(y_train) >= 0.6 or sum(y_train)/len(y_train) <= 0.4:
                smote = SMOTE(random_state=42)
                X_train_imb, y_train_imb = smote.fit_resample(X_train, y_train)
                print(f'SMOTE resampled X_train is {X_train_imb.shape}')
            else:
                X_train_imb, y_train_imb = X_train, y_train
        estimator.fit(X_train_imb, y_train_imb)
        accuracy_train.append(accuracy_score(y_train, estimator.predict(X_train)))
        precision_train.append(precision_score(y_train, estimator.predict(X_train)))
        recall_train.append(recall_score(y_train, estimator.predict(X_train)))
        auc_train.append(roc_auc_score(y_train, estimator.predict(X_train)))
        fbeta_train.append(fbeta_score(y_train, estimator.predict(X_train), beta=beta))
        accuracy_test.append(accuracy_score(y_test, estimator.predict(X_test)))
        precision_test.append(precision_score(y_test, estimator.predict(X_test)))
        recall_test.append(recall_score(y_test, estimator.predict(X_test)))
        auc_test.append(roc_auc_score(y_test, estimator.predict(X_test)))
        fbeta_test.append(fbeta_score(y_test, estimator.predict(X_test), beta=beta))
        metrics_train.append(bin_cls_evl(y_train, estimator.predict(X_train), beta=beta, print_result=False))
        metrics_test.append(bin_cls_evl(y_test, estimator.predict(X_test), beta=beta, print_result=False))
    metrics_train, metrics_test = np.array(metrics_train), np.array(metrics_test)
    scd_idx = nth_largest_idx(fbeta_test)
    # print(round(np.mean(fbeta_test), 4))
    # print()
    return(
       # np.array([
       #     np.mean(accuracy_train), np.mean(precision_train), np.mean(recall_train),
       #     np.mean(auc_train), np.mean(fbeta_train),
       #     np.mean(accuracy_test), np.mean(precision_test), np.mean(recall_test),
       #     np.mean(auc_test), np.mean(fbeta_test),
       # ]),
       # np.array([
       #     accuracy_train[scd_idx], precision_train[scd_idx], recall_train[scd_idx],
       #     auc_train[scd_idx], fbeta_train[scd_idx],
       #     accuracy_test[scd_idx], precision_test[scd_idx], recall_test[scd_idx],
       #     auc_test[scd_idx], fbeta_test[scd_idx],
       # ]),
        np.array([np.average(metrics_train, axis=0), np.average(metrics_test, axis=0),
                  metrics_train[scd_idx], metrics_test[scd_idx]])
    )

def models(X, y, estimator, n_models=50, save_path=None, save_name=None, scorer_to_print=None):
    n_samples = len(y)
    n_train = round(n_samples * 0.9)
    rand_seeds = [random.randint(1, 20 * n_models) for i in range(2*n_models)]
    scores_test = []
    for i in range(n_models):
        [X_sfl, y_sfl] = shuffle_sets([X, y], rand_seeds[:2])
        rand_seeds = rand_seeds[2:]
        X_train, X_test, y_train, y_test = X_sfl[:n_train], X_sfl[n_train:], y_sfl[:n_train], y_sfl[n_train:]
        if sum(y_train) / len(y_train) >= 0.6 or sum(y_train) / len(y_train) <= 0.4:
            smote = SMOTE(random_state=42)
            X_train_imb, y_train_imb = smote.fit_resample(X_train, y_train)
            print(f'SMOTE resampled X_train is {X_train_imb.shape}')
        else:
            X_train_imb, y_train_imb = X_train, y_train
        model = clone(estimator)
        model.fit(X_train_imb, y_train_imb)
        if scorer_to_print is not None:
            scores_test.append(scorer_to_print(y_test, model.predict(X_test)))
        if save_path is not None:
            if save_name is None:
                save_name = 'predictor'
            model_name = save_name + '_' + str(i) + '.pkl'
            save_file = os.path.join(save_path, model_name)
            joblib.dump(model, save_file)
    if scorer_to_print is not None:
        print(sum(scores_test)/n_models)
