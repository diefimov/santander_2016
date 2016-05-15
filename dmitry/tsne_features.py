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
from tsne import bh_sne
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

### add TSNE features
X = train[flist].append(test[flist], ignore_index=True).values.astype('float64')
svd = TruncatedSVD(n_components=30)
X_svd = svd.fit_transform(X)
X_scaled = StandardScaler().fit_transform(X_svd)
feats_tsne = bh_sne(X_scaled)
feats_tsne = pd.DataFrame(feats_tsne, columns=['tsne1', 'tsne2'])
feats_tsne['ID'] = train[['ID']].append(test[['ID']], ignore_index=True)['ID'].values
train = pd.merge(train, feats_tsne, on='ID', how='left')
test = pd.merge(test, feats_tsne, on='ID', how='left')

feat = train[['ID', 'tsne1', 'tsne2']].append(test[['ID', 'tsne1', 'tsne2']], ignore_index=True)
feat.to_csv(OUTPUT_PATH + 'tsne_feats.csv', index=False)
