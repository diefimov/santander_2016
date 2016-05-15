from itertools import combinations

import pandas as pd
import numpy as np
from san_utils import *

dmitry_drop = ['saldo_medio_var13_medio_ult1', 'delta_imp_reemb_var13_1y3', 'delta_imp_reemb_var17_1y3',
               'delta_imp_reemb_var33_1y3', 'delta_imp_trasp_var17_in_1y3', 'delta_imp_trasp_var17_out_1y3',
               'delta_imp_trasp_var33_in_1y3', 'delta_imp_trasp_var33_out_1y3']


kaza_drop = ["ind_var2_0","ind_var2","ind_var27_0","ind_var28_0","ind_var28","ind_var27",
"ind_var41","ind_var46_0","ind_var46","num_var27_0","num_var28_0","num_var28","num_var27","num_var41","num_var46_0",
"num_var46","saldo_var28","saldo_var27","saldo_var41","saldo_var46","imp_amort_var18_hace3","imp_amort_var34_hace3",
"imp_reemb_var13_hace3","imp_reemb_var33_hace3","imp_trasp_var17_out_hace3","imp_trasp_var33_out_hace3",
"num_var2_0_ult1","num_var2_ult1","num_reemb_var13_hace3","num_reemb_var33_hace3","num_trasp_var17_out_hace3",
"num_trasp_var33_out_hace3","saldo_var2_ult1","saldo_medio_var13_medio_hace3","ind_var6_0","ind_var6",
"ind_var13_medio_0","ind_var18_0","ind_var26_0","ind_var25_0","ind_var32_0","ind_var34_0","ind_var37_0",
"ind_var40","num_var6_0","num_var6","num_var13_medio_0","num_var18_0","num_var26_0","num_var25_0","num_var32_0",
"num_var34_0","num_var37_0","num_var40","saldo_var6","saldo_var13_medio","delta_imp_reemb_var13_1y3",
"delta_imp_reemb_var17_1y3","delta_imp_reemb_var33_1y3","delta_imp_trasp_var17_in_1y3","delta_imp_trasp_var17_out_1y3",
"delta_imp_trasp_var33_in_1y3","delta_imp_trasp_var33_out_1y3"]


drop = np.union1d(dmitry_drop, kaza_drop)
print len(drop)

train = pd.read_csv('{0}/train.csv'.format(DATA_DIR))
test = pd.read_csv('{0}/test.csv'.format(DATA_DIR))

target = train['TARGET']
train_ids = train.ID
test_ids = test.ID

train = train.drop(['ID', 'TARGET'], axis=1)
test = test.drop('ID', axis=1)

train = train.drop(drop, axis=1)
test = test.drop(drop, axis=1)

print train.shape, test.shape

ntrain = train.shape[0]
train_test = pd.concat((train, test), axis=0).reset_index(drop=True)

print "NANs:", train_test.isnull().values.any()

features = train_test.columns
print "#FEATURES", len(features)

drop = []
for feat in features:
    if np.unique(train_test[feat]).shape[0] == 1:
        print feat
        drop += [feat]

features = np.setdiff1d(features, drop)
print "#FEATURES", len(features)

drop = []
for i, pair in enumerate(combinations(features, 2)):
    f1 = pair[0]
    f2 = pair[1]
    # if f1 in drop or f2 in drop:
    #     continue
    if (train_test[f1] == train_test[f2]).all():
        print pair
        drop += [f2]

features = np.setdiff1d(features, drop)
print "#FEATURES" , len(features)

train_test = train_test[features]
print train_test.shape

train_test['zero_count'] = train_test.apply(lambda x: np.sum(x == 0), axis=1)
print train_test['zero_count'].head(n=20)

train = train_test.iloc[:ntrain,:].copy().reset_index(drop=True)
test = train_test.iloc[ntrain:,:].copy().reset_index(drop=True)

train = pd.concat((train_ids,train,target),axis=1)
test = pd.concat((test_ids,test), axis=1)

print train.shape, test.shape

train.to_csv("{0}/train_mm2.csv".format(DATA_DIR), index=None)
test.to_csv("{0}/test_mm2.csv".format(DATA_DIR), index=None)