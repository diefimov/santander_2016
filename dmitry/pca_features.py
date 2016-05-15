import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
np.random.seed(12324)
from sklearn.cross_validation import StratifiedKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from ml_metrics import auc
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from santander_preprocess import *

INPUT_PATH = '../data/input/'
OUTPUT_PATH = '../data/output/features/'

train = pd.read_csv(INPUT_PATH + 'train.csv')
test = pd.read_csv(INPUT_PATH + 'test.csv')

train, test = process_base(train, test)
train, test = drop_sparse(train, test)
train, test = drop_duplicated(train, test)
train, test = add_features(train, test, ['SumZeros'])

flist = [x for x in train.columns if not x in ['ID','TARGET']]

pca = PCA(n_components=2)
x_train_projected = pca.fit_transform(normalize(train[flist], axis=0))
x_test_projected = pca.transform(normalize(test[flist], axis=0))
train.insert(1, 'PCAOne', x_train_projected[:, 0])
train.insert(1, 'PCATwo', x_train_projected[:, 1])
test.insert(1, 'PCAOne', x_test_projected[:, 0])
test.insert(1, 'PCATwo', x_test_projected[:, 1])
pca_feats = train[['ID', 'PCAOne', 'PCATwo']].append(test[['ID', 'PCAOne', 'PCATwo']], ignore_index=True)
pca_feats.to_csv(OUTPUT_PATH + 'dmitry_pca_feats.csv')
