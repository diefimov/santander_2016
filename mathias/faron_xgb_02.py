from __future__ import division

import ntpath
import os
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score

from kaggletils import CrossValidator
from san_utils import *

__version__ = '1.0.0'

# relative path to input data (TRAIN_FILE, TEST_FILE & sample_submission.csv):
DATA_DIR = "../data/input/"
# relative path to output folder :
OUT_DIR = "./xgb02/"
ID_COLUMN = 'ID'
TARGET_COLUMN = 'TARGET'
NTRAIN = 76020
NTEST = 75818

TRAIN_FILE = 'train_mm2.csv'
TEST_FILE = 'test_mm2.csv'
TARGET_FILE = 'target.csv'

USE_AUTO_NAMING = True  # use python script name as basename for output files?
CLFNAME_PREFIX = "faron"
CLFNAME_BASE = "xgb"
CLF_VERSION = 2

if USE_AUTO_NAMING:
    CLFNAME_PREFIX = ''
    CLFNAME_BASE = ntpath.basename(__file__).replace('.py', '')
    CLF_VERSION = ''
else:
    CLFNAME_PREFIX = '{0}_'.format(CLFNAME_PREFIX)
    CLF_VERSION = '_{0:03d}'.format(CLF_VERSION)

CLF_NAME = "{0}{1}{2}".format(CLFNAME_PREFIX, CLFNAME_BASE, CLF_VERSION)

NFEATS = -1
NTHREADS = 22
SEED = 15
NBAGS = 15
AVERAGE_OOF = True  # averaging 5-Fold test predictions?
SHUFFLE_DATA = False
WRITE_SUBMISSION = False
NPOS_SCALE = 2


class XGBWrapper(object):
    def __init__(self, params=None, seed=0):
        self.xgb_params = params.copy()
        self.xgb_params['seed'] = seed
        self.nrounds = self.xgb_params.pop('nrounds', None)
        self.verbose_eval = self.xgb_params.pop('verbose_eval', False)
        self.esr = self.xgb_params.pop('esr', None)
        assert self.nrounds is not None

    def train(self, x_train, y_train, x_valid=None, y_valid=None, sample_weights=None):
        neg_ix = np.where(y_train == 0)[0]
        pos_ix = np.where(y_train == 1)[0]
        npos = pos_ix.shape[0]
        nneg = npos * NPOS_SCALE
        ix = np.sort(np.append(pos_ix, np.random.choice(neg_ix, nneg, replace=False)))

        x_train = x_train[ix]
        y_train = y_train[ix]

        dtrain = xgb.DMatrix(x_train, label=y_train)
        if x_valid is not None:
            dvalid = xgb.DMatrix(x_valid, label=y_valid)
            watchlist = [(dtrain, 'train'), (dvalid, 'val')]
            self.gbdt = xgb.train(self.xgb_params, dtrain, self.nrounds,
                                  watchlist,
                                  early_stopping_rounds=self.esr,
                                  verbose_eval=self.verbose_eval)
        else:
            self.gbdt = xgb.train(self.xgb_params, dtrain, self.nrounds)

    def predict(self, x):
        return self.gbdt.predict(xgb.DMatrix(x))


def check_folders():
    assert os.path.isdir(DATA_DIR)
    assert os.path.isdir(OUT_DIR)


def get_data():
    train = pd.read_csv("{0}/{1}".format(DATA_DIR, TRAIN_FILE))
    test = pd.read_csv("{0}/{1}".format(DATA_DIR, TEST_FILE))

    y_train = train[TARGET_COLUMN].ravel()

    features = list(train.columns)
    features = np.setdiff1d(features, [ID_COLUMN, TARGET_COLUMN])

    x_train = train[features].copy()
    x_test = test[features].copy()

    x_train = np.array(x_train)
    x_test = np.array(x_test)

    return x_train, y_train, x_test


xgb_params = {
    'seed': SEED,
    'colsample_bytree': 0.7,
    'silent': 1,
    'subsample': 0.8,
    'learning_rate': 0.02,
    'objective': "binary:logitraw",
    'max_depth': 5,
    'num_parallel_tree': 15,
    'nthread': NTHREADS,
    'min_child_weight': 3,
    'max_delta_step': 0,
    'alpha': 0,
    'lambda': 1,
    'gamma': 1,
    'eval_metric': 'auc',
    'verbose_eval': False,
    'nrounds': 400,
    'esr': None,
}

CLF_WRAPPER = XGBWrapper
CLF_PARAMS = xgb_params

if __name__ == "__main__":
    ts = datetime.now()

    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    check_folders()

    for i in range(20):
        print 'CV-SET: {0}'.format(i + 1)
        kfolds = get_kfolds(i + 1)

        cross_validator = CrossValidator(CLF_WRAPPER, CLF_PARAMS, seed=SEED,
                                         folds=kfolds, nbags=NBAGS,
                                         subsample=1, metric=roc_auc_score, shuffle=SHUFFLE_DATA,
                                         average_oof=AVERAGE_OOF, verbose=True)

        x_train, y_train, x_test = get_data()
        print x_train.shape, x_test.shape

        assert x_train.shape[0] == NTRAIN
        assert y_train.shape[0] == NTRAIN
        assert x_test.shape[0] == NTEST
        assert x_train.shape[1] == x_test.shape[1]
        assert not np.isnan(np.min(x_train))
        assert not np.isnan(np.min(y_train))
        assert not np.isnan(np.min(x_test))

        cross_validator.run_cv(x_train, y_train, x_test)
        cross_validator.print_cv_summary()

        train_file = '{0}/{1}.set{2:02d}.train.csv'.format(OUT_DIR, CLF_NAME, i + 1)
        test_file = '{0}/{1}.set{2:02d}.test.csv'.format(OUT_DIR, CLF_NAME, i + 1)
        stats_file = '{0}/{1}.set{2:02d}.stats.csv'.format(OUT_DIR, CLF_NAME, i + 1)
        sub_file = '{0}/{1}.set{2:02d}.sub.csv'.format(OUT_DIR, CLF_NAME, i + 1)

        df_oof_train = pd.DataFrame(cross_validator.oof_train, index=None)
        df_oof_test = pd.DataFrame(cross_validator.oof_test, index=None)

        df_oof_train.to_csv(train_file, index=None, header=False)
        print "{0} has been written".format(train_file)
        df_oof_test.to_csv(test_file, index=None, header=False)
        print "{0} has been written".format(test_file)

        cv_scores = cross_validator.cv_scores

        df_oof_stats = pd.DataFrame()
        df_oof_stats['Model'] = [CLF_NAME]
        df_oof_stats['Seed'] = [SEED]
        df_oof_stats['AbsMeansDelta'] = [np.abs(cross_validator.mean_train - cross_validator.mean_test)]
        df_oof_stats['CV-Mean'] = [cross_validator.cv_mean]
        df_oof_stats['CV-Std'] = [cross_validator.cv_std]
        for k, scr in enumerate(cv_scores):
            df_oof_stats['Fold {0}'.format(k + 1)] = [scr]

        df_oof_stats.to_csv(stats_file, index=None)
        print "{0} has been written".format(stats_file)

        if WRITE_SUBMISSION:
            submission = pd.read_csv('{0}/sample_submission.csv'.format(DATA_DIR))
            submission.iloc[:, 1] = cross_validator.test_predictions
            submission.to_csv(sub_file, index=None)
            print "{0} has been written".format(sub_file)

        te = datetime.now()
        print 'Overall Runtime: {0}'.format((te - ts))
