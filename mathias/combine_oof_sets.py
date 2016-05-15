import glob
from itertools import combinations

import ntpath
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from san_utils import *

OUT_DIR_TRAIN = "../data/output/train/"
OUT_DIR_TEST = "../data/output/test/"


def get_oof(subfolder=''):
    train_oof = []
    test_oof = []

    for file in glob.glob("./{0}/*.csv".format(subfolder)):
        if 'train' in file:
            train_oof.append(file)
        elif 'test' in file:
            test_oof.append(file)
        else:
            continue
    train_oof = np.sort(train_oof)
    test_oof = np.sort(test_oof)
    oof = [(a,b) for a,b in zip(train_oof,test_oof)]
    return oof


def combine_xgb_preds(model):
    train = pd.read_csv('{0}/train.csv'.format(DATA_DIR), usecols=['ID'])
    test = pd.read_csv('{0}/test.csv'.format(DATA_DIR), usecols=['ID'])

    oof = get_oof(model)
    k = 0
    for tr, te in oof:
        a = ntpath.basename(tr.replace('.train.csv', ''))
        b = ntpath.basename(te.replace('.test.csv', ''))
        assert a == b
        print a

        tr_data = pd.concat((train.ID, pd.read_csv(tr, header=None)), axis=1, ignore_index=True)
        te_data = pd.concat((test.ID, pd.read_csv(te, header=None)), axis=1, ignore_index=True)

        assert tr_data.shape[0] == train.shape[0]
        assert tr_data.shape[1] == 2
        assert te_data.shape[0] == test.shape[0]
        assert te_data.shape[1] == 2

        tr_data.columns = te_data.columns = ['ID', 'TARGET']

        meta_feat = 'set{0:02d}'.format(k+1)
        k += 1

        train = train.merge(tr_data, how='left', on='ID').rename(columns={'TARGET': meta_feat})
        test = test.merge(te_data, how='left', on='ID').rename(columns={'TARGET': meta_feat})

    yhat_train = MinMaxScaler().fit_transform(train.drop(['ID'], axis=1).mean(1).reshape(-1, 1))
    yhat_test = MinMaxScaler().fit_transform(test.drop(['ID'], axis=1).mean(1).reshape(-1, 1))

    return yhat_train, yhat_test

train = pd.read_csv('{0}/train.csv'.format(DATA_DIR), usecols=['ID'])
test = pd.read_csv('{0}/test.csv'.format(DATA_DIR), usecols=['ID'])

yhat_train_xgb1, yhat_test_xgb1 = combine_xgb_preds('xgb01')
train['faron_xgb_01'] = yhat_train_xgb1
test['faron_xgb_01'] = yhat_test_xgb1

yhat_train_xgb2, yhat_test_xgb2 = combine_xgb_preds('xgb02')
train['faron_xgb_02'] = yhat_train_xgb2
test['faron_xgb_02'] = yhat_test_xgb2

train.to_csv("{0}/faron.train.csv".format(OUT_DIR_TRAIN), index=None)
test.to_csv("{0}/faron.test.csv".format(OUT_DIR_TEST), index=None)

