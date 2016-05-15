library(dplyr)
library(glmnet)
library(foreach)
library(doSNOW)


train <- read.csv("data/input/train.csv")
idx <- read.csv("data/input/5fold_20times.csv")
train=cbind(train,idx)
test <- read.csv("data/input/test.csv")


test$TARGET=NA
test$set1=NA;test$set2=NA;test$set3=NA;test$set4=NA;test$set5=NA;
test$set6=NA;test$set7=NA;test$set8=NA;test$set9=NA;test$set10=NA;
test$set11=NA;test$set12=NA;test$set13=NA;test$set14=NA;test$set15=NA;
test$set16=NA;test$set17=NA;test$set18=NA;test$set19=NA;test$set20=NA;

full=rbind(train,test)


########################

#### DATA CLEANING #####

########################

#remove indicators - unnecessary
full=full[!names(full) %in% 
            c('ind_var1_0','ind_var1',
              'ind_var2_0','ind_var2',
              'ind_var5_0','ind_var5',
              'ind_var6_0','ind_var6',
              'ind_var7_emit_ult1','ind_var7_recib_ult1',
              'ind_var8_0','ind_var8',
              'ind_var12_0','ind_var12',
              'ind_var13_0','ind_var13',
              'ind_var13_corto_0','ind_var13_corto',
              'ind_var13_medio_0','ind_var13_medio',              
              'ind_var13_largo_0','ind_var13_largo',
              'ind_var14_0','ind_var14',
              'ind_var17_0','ind_var17',
              'ind_var18_0','ind_var18',
              'ind_var20_0','ind_var20',              
              'ind_var24_0','ind_var24',              
              'ind_var25_0','ind_var25',         
              'ind_var26_0','ind_var26',
              'ind_var27_0','ind_var27',
              'ind_var28_0','ind_var28',
              'ind_var29_0','ind_var29',
              'ind_var30_0','ind_var30',
              'ind_var31_0','ind_var31',
              'ind_var32_0','ind_var32',
              'ind_var33_0','ind_var33',
              'ind_var34_0','ind_var34',
              'ind_var37_0','ind_var37',
              'ind_var39_0','ind_var39',
              'ind_var40_0','ind_var40',
              'ind_var41_0','ind_var41',
              'ind_var43_emit_ult1','ind_var43_recib_ult1',
              'ind_var44_0','ind_var44',
              'ind_var46_0','ind_var46')]
##### remove constant features
for (f in names(full)) {  if (length(unique(full[[f]])) == 1) { print(f); full[[f]] <- NULL } }


##### if non zero column levels all TARGET=0, join information value and remove involved columns
target0cols=c() 
for (f in names(full)) {  if (max(full$TARGET[full[[f]]!=0],na.rm=T) == 0) target0cols=c(target0cols,f) }
full$s0=apply(full[,target0cols],1,function(x) sum(ifelse(x!=0,1,0)))
full$s0[full$s0>0]=1
full=full[!names(full) %in% target0cols]


##ignore errors##

##### non-0 attributes have only 1 target
target1cols=c()
for (f in setdiff(names(full),'s0')) {  try(if(table(full[[f]]==0,full$TARGET)['TRUE','1']==3007) { print(f); target1cols=c(target1cols,f) } ) }
full$s1=apply(full[,target1cols],1,function(x) sum(ifelse(x!=0,1,0)))
full=full[!names(full) %in% target1cols]

##### non-0 attributes have only 2 targets
target2cols=c()
for (f in setdiff(names(full),c('s0','s1'))) {  try(if(table(full[[f]]==0,full$TARGET)['TRUE','1']==3006) { print(f); target2cols=c(target2cols,f) } ) }
full$s2=apply(full[,target2cols],1,function(x) sum(ifelse(x!=0,1,0)))
full=full[!names(full) %in% target2cols]

##### non-0 attributes have only 3 targets
target3cols=c()
for (f in setdiff(names(full),c('s0','s1','s2'))) {  try(if(table(full[[f]]==0,full$TARGET)['TRUE','1']==3005) { print(f); target3cols=c(target3cols,f) } ) }
full$s3=apply(full[,target3cols],1,function(x) sum(ifelse(x!=0,1,0)))
full=full[!names(full) %in% target3cols]




##############################

#### FEATURE ENGINEERING #####

##############################

full$var38[full$var38==117310.979016494]=NA
full$var15[full$var15<=22]=22
full$var15[full$var15>=95]=95



########################################################
####calculate percentile rank of v38 within age group###
########################################################

###ecdf within same age group
fecdf=select(full,ID,var15,var38) %>% filter(!is.na(full$var38))
fecdf=fecdf %>% group_by(var15) %>% mutate(rank=dense_rank(var38)) %>% mutate(var38.ecdf=(rank/max(rank))) %>% data.frame()
fecdf=select(fecdf,ID,var38.ecdf)
full=left_join(full,fecdf,by=c('ID'='ID'))

auc(full$TARGET[!is.na(full$TARGET) & !is.na(full$var38)],full$var38[!is.na(full$TARGET) & !is.na(full$var38)]) #0.3983005
auc(full$TARGET[!is.na(full$TARGET) & !is.na(full$var38)],full$var38.ecdf[!is.na(full$TARGET) & !is.na(full$var38)]) #0.402414


###ecdf within age +/-1
fecdf2=select(full,ID,var15,var38) %>% mutate(var15_o=var15) %>% filter(!is.na(full$var38))
fecdf2_m1=fecdf2
fecdf2_p1=fecdf2
fecdf2_m1$var15=fecdf2_m1$var15-1
fecdf2_p1$var15=fecdf2_p1$var15+1
fecdf2=rbind(fecdf2,fecdf2_m1,fecdf2_p1)
fecdf2=fecdf2 %>% group_by(var15) %>% mutate(rank=dense_rank(var38)) %>% mutate(var38.ecdf2=(rank/max(rank))) %>% filter(var15==var15_o) %>% data.frame()
fecdf2=select(fecdf2,ID,var38.ecdf2)
full=left_join(full,fecdf2,by=c('ID'='ID'))

auc(full$TARGET[!is.na(full$TARGET) & !is.na(full$var38)],full$var38.ecdf2[!is.na(full$TARGET) & !is.na(full$var38)])

###ecdf within age +/-2
fecdf3=select(full,ID,var15,var38) %>% mutate(var15_o=var15) %>% filter(!is.na(full$var38))
fecdf3_m1=fecdf3
fecdf3_p1=fecdf3
fecdf3_m2=fecdf3
fecdf3_p2=fecdf3
fecdf3_m1$var15=fecdf3_m1$var15-1
fecdf3_p1$var15=fecdf3_p1$var15+1
fecdf3_m2$var15=fecdf3_m2$var15-2
fecdf3_p2$var15=fecdf3_p2$var15+2
fecdf3=rbind(fecdf3,fecdf3_m1,fecdf3_m2,fecdf3_p1,fecdf3_p2)
fecdf3=fecdf3 %>% group_by(var15) %>% mutate(rank=dense_rank(var38)) %>% mutate(var38.ecdf3=(rank/max(rank))) %>% filter(var15==var15_o) %>% data.frame()
fecdf3=select(fecdf3,ID,var38.ecdf3)
full=left_join(full,fecdf3,by=c('ID'='ID'))

auc(full$TARGET[!is.na(full$TARGET) & !is.na(full$var38)],full$var38.ecdf3[!is.na(full$TARGET) & !is.na(full$var38)])



###ecdf within same age group + var36
fecdf4=select(full,ID,var15,var38,var36) %>% filter(!is.na(full$var38))
fecdf4=fecdf4 %>% group_by(var15,var36) %>% mutate(rank=dense_rank(var38)) %>% mutate(var38.ecdf4=(rank/max(rank))) %>% data.frame()
fecdf4=select(fecdf4,ID,var38.ecdf4)
full=left_join(full,fecdf4,by=c('ID'='ID'))

auc(full$TARGET[!is.na(full$TARGET) & !is.na(full$var38)],full$var38[!is.na(full$TARGET) & !is.na(full$var38)]) #0.3983005
auc(full$TARGET[!is.na(full$TARGET) & !is.na(full$var38)],full$var38.ecdf4[!is.na(full$TARGET) & !is.na(full$var38)]) #0.4105126



#####################
#### X modulus 3 ####
#####################

full$imp_ent_var16_ult1r=ifelse(full$imp_ent_var16_ult1%%3==0,-1,1);full$imp_ent_var16_ult1r[full$imp_ent_var16_ult1==0]=0
full$imp_op_var39_comer_ult1r=ifelse(full$imp_op_var39_comer_ult1%%3==0,-1,1);full$imp_op_var39_comer_ult1r[full$imp_op_var39_comer_ult1==0]=0
full$imp_op_var39_comer_ult3r=ifelse(full$imp_op_var39_comer_ult3%%3==0,-1,1);full$imp_op_var39_comer_ult3r[full$imp_op_var39_comer_ult3==0]=0
full$imp_op_var40_comer_ult1r=ifelse(full$imp_op_var40_comer_ult1%%3==0,-1,1);full$imp_op_var40_comer_ult1r[full$imp_op_var40_comer_ult1==0]=0
full$imp_op_var40_comer_ult3r=ifelse(full$imp_op_var40_comer_ult3%%3==0,-1,1);full$imp_op_var40_comer_ult3r[full$imp_op_var40_comer_ult3==0]=0
full$imp_op_var40_ult1r=ifelse(full$imp_op_var40_ult1%%3==0,-1,1);full$imp_op_var40_ult1r[full$imp_op_var40_ult1==0]=0
full$imp_op_var41_comer_ult1r=ifelse(full$imp_op_var41_comer_ult1%%3==0,-1,1);full$imp_op_var41_comer_ult1r[full$imp_op_var41_comer_ult1==0]=0
full$imp_op_var41_comer_ult3r=ifelse(full$imp_op_var41_comer_ult3%%3==0,-1,1);full$imp_op_var41_comer_ult3r[full$imp_op_var41_comer_ult3==0]=0
full$imp_op_var41_efect_ult1r=ifelse(full$imp_op_var41_efect_ult1%%3==0,-1,1);full$imp_op_var41_efect_ult1r[full$imp_op_var41_efect_ult1==0]=0
full$imp_op_var41_efect_ult3r=ifelse(full$imp_op_var41_efect_ult3%%3==0,-1,1);full$imp_op_var41_efect_ult3r[full$imp_op_var41_efect_ult3==0]=0
full$imp_op_var41_ult1r=ifelse(full$imp_op_var41_ult1%%3==0,-1,1);full$imp_op_var41_ult1r[full$imp_op_var41_ult1==0]=0
full$imp_op_var39_efect_ult1r=ifelse(full$imp_op_var39_efect_ult1%%3==0,-1,1);full$imp_op_var39_efect_ult1r[full$imp_op_var39_efect_ult1==0]=0
full$imp_op_var39_efect_ult3r=ifelse(full$imp_op_var39_efect_ult3%%3==0,-1,1);full$imp_op_var39_efect_ult3r[full$imp_op_var39_efect_ult3==0]=0
full$imp_op_var39_ult1r=ifelse(full$imp_op_var39_ult1%%3==0,-1,1);full$imp_op_var39_ult1r[full$imp_op_var39_ult1==0]=0
full$imp_sal_var16_ult1r=ifelse(full$imp_sal_var16_ult1%%3==0,-1,1);full$imp_sal_var16_ult1r[full$imp_sal_var16_ult1==0]=0
full$saldo_var1r=ifelse(full$saldo_var1%%3==0,-1,1);full$saldo_var1r[full$saldo_var1==0]=0
full$saldo_var5r=ifelse(full$saldo_var5%%3==0,-1,1);full$saldo_var5r[full$saldo_var5==0]=0
full$saldo_var8r=ifelse(full$saldo_var8%%3==0,-1,1);full$saldo_var8r[full$saldo_var8==0]=0
full$saldo_var12r=ifelse(full$saldo_var12%%3==0,-1,1);full$saldo_var12r[full$saldo_var12==0]=0
full$saldo_var13_cortor=ifelse(full$saldo_var13_corto%%3==0,-1,1);full$saldo_var13_cortor[full$saldo_var13_corto==0]=0
full$saldo_var13r=ifelse(full$saldo_var13%%3==0,-1,1);full$saldo_var13r[full$saldo_var13==0]=0
full$saldo_var14r=ifelse(full$saldo_var14%%3==0,-1,1);full$saldo_var14r[full$saldo_var14==0]=0
full$saldo_var24r=ifelse(full$saldo_var24%%3==0,-1,1);full$saldo_var24r[full$saldo_var24==0]=0
full$saldo_var25r=ifelse(full$saldo_var25%%3==0,-1,1);full$saldo_var25r[full$saldo_var25==0]=0
full$saldo_var26r=ifelse(full$saldo_var26%%3==0,-1,1);full$saldo_var26r[full$saldo_var26==0]=0
full$saldo_var30r=ifelse(full$saldo_var30%%3==0,-1,1);full$saldo_var30r[full$saldo_var30==0]=0
full$saldo_var31r=ifelse(full$saldo_var31%%3==0,-1,1);full$saldo_var31r[full$saldo_var31==0]=0
full$saldo_var37r=ifelse(full$saldo_var37%%3==0,-1,1);full$saldo_var37r[full$saldo_var37==0]=0
full$saldo_var40r=ifelse(full$saldo_var40%%3==0,-1,1);full$saldo_var40r[full$saldo_var40==0]=0
full$saldo_var42r=ifelse(full$saldo_var42%%3==0,-1,1);full$saldo_var42r[full$saldo_var42==0]=0
full$imp_aport_var13_hace3r=ifelse(full$imp_aport_var13_hace3%%3==0,-1,1);full$imp_aport_var13_hace3r[full$imp_aport_var13_hace3==0]=0
full$imp_aport_var13_ult1r=ifelse(full$imp_aport_var13_ult1%%3==0,-1,1);full$imp_aport_var13_ult1r[full$imp_aport_var13_ult1==0]=0
full$imp_var7_recib_ult1r=ifelse(full$imp_var7_recib_ult1%%3==0,-1,1);full$imp_var7_recib_ult1r[full$imp_var7_recib_ult1==0]=0
full$imp_var43_emit_ult1r=ifelse(full$imp_var43_emit_ult1%%3==0,-1,1);full$imp_var43_emit_ult1r[full$imp_var43_emit_ult1==0]=0
full$imp_trans_var37_ult1r=ifelse(full$imp_trans_var37_ult1%%3==0,-1,1);full$imp_trans_var37_ult1r[full$imp_trans_var37_ult1==0]=0
full$saldo_medio_var5_hace2r=ifelse(full$saldo_medio_var5_hace2%%3==0,-1,1);full$saldo_medio_var5_hace2r[full$saldo_medio_var5_hace2==0]=0                  
full$saldo_medio_var5_hace3r=ifelse(full$saldo_medio_var5_hace3%%3==0,-1,1);full$saldo_medio_var5_hace3r[full$saldo_medio_var5_hace3==0]=0  
full$saldo_medio_var5_ult1r=ifelse(full$saldo_medio_var5_ult1%%3==0,-1,1);full$saldo_medio_var5_ult1r[full$saldo_medio_var5_ult1==0]=0  
full$saldo_medio_var5_ult3r=ifelse(full$saldo_medio_var5_ult3%%3==0,-1,1);full$saldo_medio_var5_ult3r[full$saldo_medio_var5_ult3==0]=0  
full$saldo_medio_var8_hace2r=ifelse(full$saldo_medio_var8_hace2%%3==0,-1,1);full$saldo_medio_var8_hace2r[full$saldo_medio_var8_hace2==0]=0  
full$saldo_medio_var8_hace3r=ifelse(full$saldo_medio_var8_hace3%%3==0,-1,1);full$saldo_medio_var8_hace3r[full$saldo_medio_var8_hace3==0]=0  
full$saldo_medio_var8_ult1r=ifelse(full$saldo_medio_var8_ult1%%3==0,-1,1);full$saldo_medio_var8_ult1r[full$saldo_medio_var8_ult1==0]=0  
full$saldo_medio_var8_ult3r=ifelse(full$saldo_medio_var8_ult3%%3==0,-1,1);full$saldo_medio_var8_ult3r[full$saldo_medio_var8_ult3==0]=0 
full$saldo_medio_var12_hace2r=ifelse(full$saldo_medio_var12_hace2%%3==0,-1,1);full$saldo_medio_var12_hace2r[full$saldo_medio_var12_hace2==0]=0 
full$saldo_medio_var12_hace3r=ifelse(full$saldo_medio_var12_hace3%%3==0,-1,1);full$saldo_medio_var12_hace3r[full$saldo_medio_var12_hace3==0]=0 
full$saldo_medio_var12_ult1r=ifelse(full$saldo_medio_var12_ult1%%3==0,-1,1);full$saldo_medio_var12_ult1r[full$saldo_medio_var12_ult1==0]=0 
full$saldo_medio_var12_ult3r=ifelse(full$saldo_medio_var12_ult3%%3==0,-1,1);full$saldo_medio_var12_ult3r[full$saldo_medio_var12_ult3==0]=0 
full$saldo_medio_var13_corto_hace2r=ifelse(full$saldo_medio_var13_corto_hace2%%3==0,-1,1);full$saldo_medio_var13_corto_hace2r[full$saldo_medio_var13_corto_hace2==0]=0 
full$saldo_medio_var13_corto_hace3r=ifelse(full$saldo_medio_var13_corto_hace3%%3==0,-1,1);full$saldo_medio_var13_corto_hace3r[full$saldo_medio_var13_corto_hace3==0]=0 
full$saldo_medio_var13_corto_ult1r=ifelse(full$saldo_medio_var13_corto_ult1%%3==0,-1,1);full$saldo_medio_var13_corto_ult1r[full$saldo_medio_var13_corto_ult1==0]=0 
full$saldo_medio_var13_corto_ult3r=ifelse(full$saldo_medio_var13_corto_ult3%%3==0,-1,1);full$saldo_medio_var13_corto_ult3r[full$saldo_medio_var13_corto_ult3==0]=0 


full$var38NA=0
full$var38NA[is.na(full$var38)]=1
full$var38[full$var38NA==1]=0
full$var38.ecdf[full$var38NA==1]=0
full$var38.ecdf2[full$var38NA==1]=0
full$var38.ecdf3[full$var38NA==1]=0
full$var38.ecdf4[full$var38NA==1]=0


full=cbind(full,model.matrix(~-1+as.factor(var15),data=full))


feature.names=setdiff(names(full),c('ID','TARGET','FOLD',
                                    'set1','set2','set3','set4','set5',
                                    'set6','set7','set8','set9','set10',
                                    'set11','set12','set13','set14','set15',
                                    'set16','set17','set18','set19','set20'
))
for (f in feature.names) {
  full[[f]] = log(full[[f]] - min(full[[f]]) + 1)
}



cl <- makeCluster(20, type="SOCK")
registerDoSNOW(cl)

preds.A=list()

lambda=0.000344356

preds=foreach (i = 1:20) %dopar% {
  
  require(glmnet)
  require(dplyr)
  
  set=full[[names(full)[grep('set',names(full))][i]]]

  train.glmnet=data.matrix(full[!is.na(full$TARGET),feature.names])
  train.glmnet.y=as.matrix(full$TARGET[!is.na(full$TARGET)])
  train.glmnet0=data.matrix(full[!is.na(full$TARGET) & set!=0,feature.names])
  train.glmnet0.y=as.matrix(full$TARGET[!is.na(full$TARGET) & set!=0])
  train.glmnet1=data.matrix(full[!is.na(full$TARGET) & set!=1,feature.names])
  train.glmnet1.y=as.matrix(full$TARGET[!is.na(full$TARGET) & set!=1])
  train.glmnet2=data.matrix(full[!is.na(full$TARGET) & set!=2,feature.names])
  train.glmnet2.y=as.matrix(full$TARGET[!is.na(full$TARGET) & set!=2])
  train.glmnet3=data.matrix(full[!is.na(full$TARGET) & set!=3,feature.names])
  train.glmnet3.y=as.matrix(full$TARGET[!is.na(full$TARGET) & set!=3])
  train.glmnet4=data.matrix(full[!is.na(full$TARGET) & set!=4,feature.names])
  train.glmnet4.y=as.matrix(full$TARGET[!is.na(full$TARGET) & set!=4])
  train.glmnet=data.matrix(full[!is.na(full$TARGET),feature.names])
  train.glmnet.y=as.matrix(full$TARGET[!is.na(full$TARGET)])
  full.glmnet=data.matrix(full[,feature.names])
  
  
  set.seed(1)
  glmnet.model1=glmnet(x=train.glmnet0,y=train.glmnet0.y,family='binomial',nlambda=1000)
  glmnet.model2=glmnet(x=train.glmnet1,y=train.glmnet1.y,family='binomial',nlambda=1000)
  glmnet.model3=glmnet(x=train.glmnet2,y=train.glmnet2.y,family='binomial',nlambda=1000)
  glmnet.model4=glmnet(x=train.glmnet3,y=train.glmnet3.y,family='binomial',nlambda=1000)
  glmnet.model5=glmnet(x=train.glmnet4,y=train.glmnet4.y,family='binomial',nlambda=1000)
  
  full$glmnet.pred1_1=predict(object=glmnet.model1,newx=full.glmnet,s=lambda, type='response')[,1]
  full$glmnet.pred2_1=predict(object=glmnet.model2,newx=full.glmnet,s=lambda, type='response')[,1]
  full$glmnet.pred3_1=predict(object=glmnet.model3,newx=full.glmnet,s=lambda, type='response')[,1]
  full$glmnet.pred4_1=predict(object=glmnet.model4,newx=full.glmnet,s=lambda, type='response')[,1]
  full$glmnet.pred5_1=predict(object=glmnet.model5,newx=full.glmnet,s=lambda, type='response')[,1]
  
  
  full$glmnet.pred=NA
  full$glmnet.pred[set==0 & !is.na(set)]=(full$glmnet.pred1_1)[set==0 & !is.na(set)]
  full$glmnet.pred[set==1 & !is.na(set)]=(full$glmnet.pred2_1)[set==1 & !is.na(set)]
  full$glmnet.pred[set==2 & !is.na(set)]=(full$glmnet.pred3_1)[set==2 & !is.na(set)]
  full$glmnet.pred[set==3 & !is.na(set)]=(full$glmnet.pred4_1)[set==3 & !is.na(set)]
  full$glmnet.pred[set==4 & !is.na(set)]=(full$glmnet.pred5_1)[set==4 & !is.na(set)]
  full$glmnet.pred[is.na(set)]=(full$glmnet.pred1_1+full$glmnet.pred2_1+full$glmnet.pred3_1+full$glmnet.pred4_1+full$glmnet.pred5_1)[is.na(set)]/5
  
  
  select(full,ID,glmnet.pred)
  
}

for (i in 1:20) {
  preds[[i]]$glmnet.pred=rank(preds[[i]]$glmnet.pred)
}
full=left_join(full,Reduce('+',preds)/20,by=c('ID'='ID'))

write.csv(select(filter(full,!is.na(TARGET)),ID,glmnet.pred),'data/output/temp/lasso_raddar_train.csv',row.names=FALSE)
write.csv(select(filter(full,is.na(TARGET)),ID,glmnet.pred),'data/output/temp/lasso_raddar_test.csv',row.names=FALSE)







