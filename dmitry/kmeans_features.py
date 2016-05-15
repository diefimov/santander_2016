import numpy as np
import xgboost as xgb
import pandas as pd
from scipy.stats.stats import pearsonr
np.random.seed(12324)
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from ml_metrics import auc
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from tsne import bh_sne
import os
from sklearn.datasets import dump_svmlight_file
import scipy.sparse as sp
from sklearn.cluster import KMeans
from santander_preprocess import *

INPUT_PATH = '../data/input/'
OUTPUT_PATH = '../data/output/features/'

train = pd.read_csv(INPUT_PATH + 'train.csv')
test = pd.read_csv(INPUT_PATH + 'test.csv')

train, test = process_base(train, test)
train, test = drop_sparse(train, test)
train, test = drop_duplicated(train, test)
train, test = add_features(train, test, ['SumZeros'])
train, test = normalize_features(train, test)

flist = [x for x in train.columns if not x in ['ID','TARGET']]

flist_kmeans = []
for ncl in range(2,11):
    cls = KMeans(n_clusters=ncl)
    cls.fit_predict(train[flist].values)
    train['kmeans_cluster'+str(ncl)] = cls.predict(train[flist].values)
    test['kmeans_cluster'+str(ncl)] = cls.predict(test[flist].values)
    flist_kmeans.append('kmeans_cluster'+str(ncl))

train[['ID']+flist_kmeans].append(test[['ID']+flist_kmeans], ignore_index=True).to_csv(OUTPUT_PATH + 'kmeans_feats.csv', index=False)



