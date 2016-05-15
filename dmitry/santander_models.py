import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from rgf import RGFRegressor
import xgboost as xgb
import os
import glob

def train_predict_logistic_regression(X_train, y_train, X_test):
    clf = LogisticRegression(C = 2.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    return y_pred


def train_predict_gaussian_process(X_train, y_train, X_test):
    clf = GaussianProcess(theta0=0.1, thetaL=.001, thetaU=1.)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred


def train_predict_extra_trees(X_train, y_train, X_test):
    clf = ExtraTreesClassifier(n_estimators=1000, criterion='gini', min_samples_split=400, min_samples_leaf=1, max_features=280, n_jobs=-1, random_state=32934)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    return y_pred

def train_predict_rgf(X_train, y_train, X_test):
    clf = RGFRegressor(working_directory='../data/output-rgf/', 
                       rgf_bin = '../rgf1.2/bin/', 
                       loss = "LS",
                       reg_L2 = 0.5,
                       reg_sL2 = 0.1,
                       test_interval=500, 
                       max_leaf_forest=1000)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_many(X_test)[-1][1]
    clf.clean_files()
    return y_pred

def train_predict_adaboost_classifier(X_train, y_train, X_test):
    clf = AdaBoostClassifier(n_estimators=300, learning_rate=0.1, random_state=32934)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    return y_pred

def train_predict_xgboost(X_train, y_train, X_test):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.02
    param['max_depth'] = 5
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    param['nthread'] = 6
    param['gamma'] = 1.0
    param['min_child_weight'] = 5
    param['subsample'] = 0.8
    param['colsample_bytree'] = 1.0
    param['colsample_bylevel'] = 0.7
    num_round = 500
    param['seed'] = 123089
    plst = list(param.items())
    xgmat_train = xgb.DMatrix(X_train, label=y_train, missing = -999.0)
    xgmat_test = xgb.DMatrix(X_test, missing = -999.0)
    bst = xgb.train(plst, xgmat_train, num_round)
    y_pred = bst.predict( xgmat_test )
    return y_pred

def train_predict_xgboost_bugged(X_train, y_train, X_test):
    param = {}
    param['objective'] = 'binary:logistic'
    param['eta'] = 0.02
    param['max_depth'] = 5
    param['eval_metric'] = 'auc'
    param['silent'] = 1
    param['nthread'] = 6
    param['gamma'] = 1.0
    param['min_child_weight'] = 5
    param['subsample'] = 0.8
    param['colsample_bytree'] = 1.0
    param['colsample_bylevel'] = 0.7
    num_round = 500

    y_pred = [0.0]*len(X_test)
    for seed in [123089, 21324, 324003, 450453, 120032]:
        param['seed'] = seed
        plst = list(param.items())
        xgmat_train = xgb.DMatrix(X_train, label=y_train, missing = -999.0)
        xgmat_test = xgb.DMatrix(X_test, missing = -999.0)
        bst = xgb.train(plst, xgmat_train, num_round)
        y_pred = y_pred + bst.predict( xgmat_test )
    y_pred = y_pred/5.0
    return y_pred

def train_predict_kernel_ridge(X_train, y_train, X_test):
    clf = KernelRidge(alpha=1.0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def train_predict_adaboost_regressor(X_train, y_train, X_test):
    clf = AdaBoostRegressor(n_estimators=100, learning_rate=0.3, random_state=32934)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    return y_pred

def train_predict_bagging_classifier(X_train, y_train, X_test):
    clf = BaggingClassifier(n_estimators=100, n_jobs=-1, random_state=32934)
    clf.fit(X_train, y_train)
    y_pred = clf.predict_proba(X_test)[:,1]
    return y_pred

def train_predict_ftrl(X_train, y_train, X_test):
    train_file = '../data/output-ftrl/train_ftrl.csv'
    test_file = '../data/output-ftrl/test_ftrl.csv'
    pred_file = '../data/output-ftrl/pred_ftrl.csv'

    train_csv = pd.DataFrame(X_train)
    train_csv['TARGET'] = y_train
    train_csv['ID'] = [x for x in range(1, len(train_csv)+1)]
    train_csv.to_csv(train_file, index=False)

    test_csv = pd.DataFrame(X_test)
    test_csv['ID'] = [x for x in range(1, len(test_csv)+1)]
    test_csv.to_csv(test_file, index=False)

    non_factor_cols = "''"
    non_feature_cols = "''"
    text_cols = "''"

    os.system('pypy ftrl.py' +
              ' --alpha ' + str(0.06) +
              ' --beta ' + str(1.0) +
              ' --L1 ' + str(0.01) +
              ' --L2 ' + str(1.0) +
              ' --epoch ' + str(3) +
              ' --train ' + train_file +
              ' --test ' + test_file +
              ' --submission ' + pred_file +
              ' --non_feature_cols ' + non_feature_cols +
              ' --non_factor_cols ' + non_factor_cols + 
              ' --text_cols ' + text_cols)

    y_pred = pd.read_csv(pred_file)['PRED'].values
    filelist = glob.glob("../data/output-ftrl/*.*")
    for f in filelist:
        os.remove(f)
    return y_pred







