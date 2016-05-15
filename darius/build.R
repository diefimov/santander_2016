#setwd('~/KAGGLE/Santander/submission/')
source('scripts/population_split.R')
source('scripts/svm_raddar.R')
source('scripts/lasso_raddar.R')
source('scripts/xgb_raddar_1.R')
source('scripts/xgb_raddar_2.R')


svm.train=read.csv('data/output/temp/linearsvm_raddar_train.csv')
lasso.train=read.csv('data/output/temp/lasso_raddar_train.csv')
xgb1.train=read.csv('data/output/temp/xgb_raddar_1_train.csv')
xgb2.train=read.csv('data/output/temp/xgb_raddar_2_train.csv')

svm.test=read.csv('data/output/temp/linearsvm_raddar_test.csv')
lasso.test=read.csv('data/output/temp/lasso_raddar_test.csv')
xgb1.test=read.csv('data/output/temp/xgb_raddar_1_test.csv')
xgb2.test=read.csv('data/output/temp/xgb_raddar_2_test.csv')

X.train=svm.train
X.train=merge(X.train,lasso.train)
X.train=merge(X.train,xgb1.train)
X.train=merge(X.train,xgb2.train)

X.test=svm.test
X.test=merge(X.test,lasso.test)
X.test=merge(X.test,xgb1.test)
X.test=merge(X.test,xgb2.test)

names(X.train)=c('ID','darius_SVM.PRED.A','darius_glmnet.pred','darius_xgb_raddar1','darius_xgb_raddar2')
names(X.test)=c('ID','darius_SVM.PRED.A','darius_glmnet.pred','darius_xgb_raddar1','darius_xgb_raddar2')

write.csv(X.train,'data/output/train/raddar_models_train.csv',row.names=F)
write.csv(X.test,'data/output/test/raddar_models_test.csv',row.names=F)
