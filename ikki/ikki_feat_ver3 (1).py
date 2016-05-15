#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import os
#os.chdir('/Users/IkkiTanaka/Documents/kaggle/Santander/')

#PATH
from base_fixed_fold import FOLDER_NAME, PATH, INPUT_PATH, OUTPUT_PATH, ORIGINAL_TRAIN_FORMAT, SUBMIT_FORMAT


#import bloscpack
from datetime import date

from sklearn.manifold import TSNE

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Normalizer, PolynomialFeatures, MinMaxScaler



########### Feature Engineering ############

######### Reading data ###########
ori_train = pd.read_csv('../data/input/train.csv')
ori_test = pd.read_csv('../data/input/test.csv')
sample_submit = pd.read_csv('data/input/sample_submission.csv')

ori_train['target'] = ori_train['TARGET']
ori_train['t_id'] = ori_train["ID"]
ori_test['t_id'] = ori_test["ID"]

del ori_train['TARGET'], ori_train["ID"], ori_test["ID"]




def main_feat_NN():
    train = pd.read_csv('data/output/features/ikki_features_train_ver1.csv')
    test = pd.read_csv('data/output/features/ikki_features_test_ver1.csv')

    train_target = train['target']
    del train['target']

    #delete id
    #del train['t_id'], test['t_id']
    ohe_col = ['num_var13_corto','num_var13_corto_0','num_meses_var12_ult3','num_meses_var13_corto_ult3','num_meses_var39_vig_ult3','num_meses_var5_ult3','num_var24_0','num_var12','var36','num_var5','num_var5_0','num_var12_0','num_var13','num_var13_0','num_var42','num_var4','num_var42_0','num_var30','num_var39_0','num_var41_0']

    #delete categorical columns
    #because OHEncoder is ismplemented in another func.
    for i in ohe_col:
        del train[i], test[i]

    #delete var3 because var3 is OHEncoded in main_feature() 
    del train['var3']
    del test['var3']

    #replace min/max in test with min/max in train
    for i in train.columns:
        min_val = train[i].min()
        test.loc[(test[i] < min_val).values,i] = min_val

        max_val = train[i].max()
        test.loc[(test[i] > max_val).values,i] = max_val


    
    #log transformation
    train_test = pd.concat([train, test])
    train_test_min = train_test.min()
    train_test = train_test - train_test_min
    train = train_test.iloc[:len(train),:]
    test = train_test.iloc[len(train):,:]

    train = train.applymap(lambda x: np.log(x + 1))
    test = test.applymap(lambda x: np.log(x + 1))
    assert( all(train.columns == test.columns))
    
    train['target'] = train_target

    train.to_csv('data/output/features/ikki_features_train_NN_ver3.csv',index=None)
    test.to_csv('data/output/features/ikki_features_test_NN_ver3.csv',index=None)



if __name__ == '__main__':
    print 'Creating dataset (Feature engineering)'
    main_feat_NN()
    print 'Done dataset creation'




