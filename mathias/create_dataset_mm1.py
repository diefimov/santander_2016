from itertools import combinations

import pandas as pd
import numpy as np
from san_utils import *

drop = ['delta_imp_amort_var18_1y3', 'delta_imp_amort_var34_1y3',
'delta_imp_aport_var17_1y3', 'delta_imp_aport_var33_1y3', 'delta_imp_reemb_var13_1y3',
'delta_imp_reemb_var33_1y3', 'delta_imp_trasp_var17_in_1y3',
'delta_imp_trasp_var17_out_1y3', 'delta_imp_trasp_var33_in_1y3',
'delta_imp_trasp_var33_out_1y3', 'delta_imp_venta_var44_1y3', 'delta_num_aport_var17_1y3',
'delta_num_aport_var33_1y3', 'delta_num_compra_var44_1y3',
'delta_num_venta_var44_1y3', 'imp_amort_var18_ult1', 'imp_amort_var34_ult1',
'imp_aport_var17_hace3', 'imp_aport_var33_hace3', 'imp_aport_var33_ult1',
'imp_compra_var44_hace3', 'imp_reemb_var17_hace3', 'imp_reemb_var17_ult1',
'imp_reemb_var33_ult1', 'imp_trasp_var17_in_hace3', 'imp_trasp_var17_in_ult1',
'imp_trasp_var17_out_ult1', 'imp_trasp_var33_in_hace3', 'imp_trasp_var33_in_ult1',
'imp_trasp_var33_out_ult1', 'imp_var7_emit_ult1', 'imp_venta_var44_hace3',
'imp_venta_var44_ult1', 'ind_var13_corto', 'ind_var13_corto_0', 'ind_var13_largo_0',
'ind_var13_medio', 'ind_var18', 'ind_var20', 'ind_var20_0', 'ind_var24', 'ind_var29',
'ind_var29_0', 'ind_var32_cte', 'ind_var33', 'ind_var33_0', 'ind_var34',
'ind_var44', 'ind_var44_0', 'ind_var7_emit_ult1', 'num_aport_var17_hace3',
'num_aport_var33_hace3', 'num_aport_var33_ult1', 'num_compra_var44_hace3',
'num_compra_var44_ult1', 'num_meses_var13_largo_ult3',
'num_meses_var13_medio_ult3', 'num_meses_var29_ult3', 'num_meses_var33_ult3',
'num_meses_var44_ult3', 'num_op_var40_efect_ult1', 'num_op_var40_hace3',
'num_reemb_var13_ult1', 'num_reemb_var17_hace3', 'num_reemb_var17_ult1',
'num_reemb_var33_ult1', 'num_trasp_var17_in_hace3', 'num_trasp_var17_in_ult1',
'num_trasp_var17_out_ult1', 'num_trasp_var33_in_hace3', 'num_trasp_var33_in_ult1',
'num_trasp_var33_out_ult1', 'num_var1', 'num_var12', 'num_var13_corto', 'num_var13_corto_0',
'num_var13_largo', 'num_var13_largo_0', 'num_var13_medio', 'num_var13_medio_0',
'num_var18', 'num_var20', 'num_var20_0', 'num_var24', 'num_var29', 'num_var29_0',
'num_var31', 'num_var31_0', 'num_var32', 'num_var33', 'num_var33_0', 'num_var34',
'num_var39', 'num_var44', 'num_var44_0', 'num_var7_emit_ult1',
'num_var7_recib_ult1', 'num_venta_var44_hace3', 'num_venta_var44_ult1',
'saldo_medio_var13_largo_hace2', 'saldo_medio_var13_largo_hace3',
'saldo_medio_var13_largo_ult1', 'saldo_medio_var13_largo_ult3',
'saldo_medio_var13_medio_hace2', 'saldo_medio_var13_medio_ult1',
'saldo_medio_var13_medio_ult3', 'saldo_medio_var17_hace2', 'saldo_medio_var17_hace3',
'saldo_medio_var17_ult1', 'saldo_medio_var29_hace2', 'saldo_medio_var29_hace3',
'saldo_medio_var29_ult1', 'saldo_medio_var29_ult3', 'saldo_medio_var33_hace2',
'saldo_medio_var33_hace3', 'saldo_medio_var33_ult1', 'saldo_medio_var33_ult3',
'saldo_medio_var44_hace2', 'saldo_medio_var44_hace3', 'saldo_medio_var44_ult3',
'saldo_var13_largo', 'saldo_var13_medio', 'saldo_var18', 'saldo_var20', 'saldo_var29',
'saldo_var33', 'saldo_var34', 'delta_imp_compra_var44_1y3',
'delta_imp_reemb_var17_1y3', 'delta_num_reemb_var13_1y3', 'delta_num_reemb_var33_1y3',
'delta_num_trasp_var17_in_1y3', 'delta_num_trasp_var17_out_1y3',
'delta_num_trasp_var33_in_1y3', 'delta_num_trasp_var33_out_1y3', 'imp_aport_var17_ult1',
'imp_compra_var44_ult1', 'imp_reemb_var13_ult1', 'ind_var13_largo',
'ind_var13_medio_0', 'ind_var14', 'ind_var17', 'ind_var17_0', 'ind_var18_0', 'ind_var19',
'ind_var32', 'ind_var34_0', 'ind_var6', 'ind_var6_0', 'ind_var7_recib_ult1',
'num_aport_var17_ult1', 'num_meses_var17_ult3', 'num_op_var40_comer_ult1',
'num_op_var40_efect_ult3', 'num_op_var40_hace2', 'num_sal_var16_ult1', 'num_var14',
'num_var17', 'num_var17_0', 'num_var18_0', 'num_var32_0', 'num_var34_0',
'num_var6', 'num_var6_0', 'saldo_medio_var17_ult3',
'saldo_medio_var44_ult1', 'saldo_var17', 'saldo_var32', 'saldo_var44', 'saldo_var6',
'delta_num_reemb_var17_1y3', 'imp_op_var40_efect_ult1', 'ind_var1', 'ind_var31',
'ind_var32_0', 'ind_var39', 'num_op_var40_ult1', 'num_var40',
'imp_op_var40_comer_ult1', 'num_aport_var13_ult1', 'num_op_var40_comer_ult3',
'imp_var7_recib_ult1', 'ind_var40', 'num_op_var40_ult3', 'imp_op_var40_comer_ult3',
'imp_sal_var16_ult1', 'saldo_medio_var13_corto_hace3']

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

train.to_csv("{0}/train_mm1.csv".format(DATA_DIR), index=None)
test.to_csv("{0}/test_mm1.csv".format(DATA_DIR), index=None)