import numpy as np
import pandas as pd

NTRAIN = 76020
NTEST = 75818

DATA_DIR = "../data/input"
CVFILE = "5fold_20times.csv"
TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"


def get_cv_indices(set=1, fold=0):
    sets = pd.read_csv('{0}/{1}'.format(DATA_DIR,CVFILE))
    kfolds = sets['set{0}'.format(set)].copy()
    if fold in range(5):
        kix_train = kfolds[kfolds != fold].index
        kix_valid = kfolds[kfolds == fold].index
    else:
        kix_train = kfolds.index
        kix_valid = []
    return np.array(kix_train), np.array(kix_valid)


def get_kfolds(set=1):
    kfolds = []
    for k in range(5):
        kfolds.append(get_cv_indices(set=set, fold=k))
    return kfolds


