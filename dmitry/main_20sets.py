import numpy as np
import pandas as pd
from scipy.stats.stats import pearsonr
np.random.seed(12324)
from ml_metrics import auc
import os
from santander_preprocess import *
from santander_models import *
import sys, getopt, re

INPUT_PATH = '../data/input/'
OUTPUT_PATH = '../data/output/'

MODELS_ALL = ['ftrl2', 'rgf3', 'rgf5', 'rgf6', 'adaboost_classifier', 'xgboost']
FEATURES_ALL = [['SumZeros', 'likeli'], 
                ['SumZeros', 'pca', 'likeli'], 
                ['SumZeros', 'tsne', 'likeli'], 
                ['SumZeros', 'kmeans', 'likeli'], 
                ['SumZeros', 'pca', 'likeli'],
                ['SumZeros', 'pca', 'likeli']]

train = pd.read_csv(INPUT_PATH + 'train.csv')
test = pd.read_csv(INPUT_PATH + 'test.csv')
preds_all = train[['ID']].append(test[['ID']], ignore_index=True).copy()
for imod in range(len(MODELS_ALL)):
    MODEL = MODELS_ALL[imod]
    FEATURES = FEATURES_ALL[imod]
    print 'Training ' + MODEL + '...'

    train = pd.read_csv(INPUT_PATH + 'train.csv')
    test = pd.read_csv(INPUT_PATH + 'test.csv')
    id_fold = pd.read_csv(INPUT_PATH+'5fold_20times.csv')
    id_fold['ID'] = train['ID'].values

    train, test = process_base(train, test)
    train, test = drop_sparse(train, test)
    train, test = drop_duplicated(train, test)
    train, test = add_features(train, test, FEATURES)

    flist = [x for x in train.columns if not x in ['ID','TARGET']]

    preds_model = pd.DataFrame()
    for it in range(1, 21):
        print 'Processing iteration ' + str(it) + '...'   
        it_id_fold = id_fold[['ID', 'set'+str(it)]]
        it_id_fold.columns = ['ID', 'FOLD']
        if 'FOLD' in train.columns:
            train.drop('FOLD', axis=1, inplace=True)
        train = pd.merge(train, it_id_fold, on='ID', how='left')
        aucs = []
        for fold in range(5):
            train_split = train.query('FOLD != @fold').copy().reset_index(drop=True)
            y_train = train_split['TARGET'].values
            val_split = train.query('FOLD == @fold').copy().reset_index(drop=True)
            test_split = val_split[['ID']+flist].append(test[['ID']+flist], ignore_index=True)
            ids_val = val_split['ID'].values

            if 'likeli' in FEATURES:
                train_split, test_split, flist1 = add_likelihood_feature('saldo_var13', train_split, test_split, flist)
            else:
                flist1 = flist
            
            X_train = train_split[flist1].values
            y_train = train_split['TARGET'].values
            X_test = test_split[flist1].values

            if 'rgf' in MODEL:
                y_pred = train_predict_rgf(X_train, y_train, X_test)

            if MODEL == 'extra_trees':
                y_pred = train_predict_extra_trees(X_train, y_train, X_test)

            if MODEL == 'xgboost':
                y_pred = train_predict_xgboost_bugged(X_train, y_train, X_test)            

            if MODEL == 'adaboost_classifier':
                y_pred = train_predict_adaboost_classifier(X_train, y_train, X_test)

            if 'ftrl' in MODEL:
                y_pred = train_predict_ftrl(X_train, y_train, X_test)
            
            preds = pd.DataFrame()
            preds['ID'] = test_split['ID'].values
            preds['FOLD'] = fold
            preds['ITER'] = it
            preds[MODEL] = y_pred
            preds_model = preds_model.append(preds, ignore_index=True)

            preds = preds.loc[preds['ID'].isin(ids_val)].copy()
            preds = pd.merge(preds, train[['ID', 'TARGET']], on='ID', how='left')

            fold_auc = auc(preds['TARGET'], preds[MODEL])
            aucs.append(fold_auc)
        print np.mean(aucs), np.std(aucs)

    preds_model.loc[preds_model[MODEL]<0, MODEL] = 0.0
    preds_model.loc[preds_model[MODEL]>1, MODEL] = 1.0
    preds_model = preds_model.groupby(['ID', 'ITER'])[MODEL].mean().reset_index()
    for it in range(1, 21):
        preds_model.loc[preds_model['ITER']==it, MODEL] = preds_model.loc[preds_model['ITER']==it, MODEL].rank()
    preds_model = preds_model.groupby('ID')[MODEL].mean().reset_index()
    preds_model.columns = ['ID', 'dmitry_'+MODEL]
    preds_all = pd.merge(preds_all, preds_model, on='ID', how='left')
    preds_all.to_csv('all_models_temp.csv', index=False)

preds_train = pd.merge(train[['ID']], preds_all, on='ID', how='left')
preds_train.to_csv(OUTPUT_PATH + 'train/' + 'dmitry_train.csv', index=False)
preds_test = pd.merge(test[['ID']], preds_all, on='ID', how='left')
preds_test.to_csv(OUTPUT_PATH + 'test/' + 'dmitry_test.csv', index=False)
print "Done training!"
