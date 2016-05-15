library(dplyr)
library(Metrics)
library(xgboost)


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
full$var3[full$var3==-999999]=-1
full$delta_num_aport_var13_1y3[full$delta_num_aport_var13_1y3==9999999999]=10
full$delta_imp_aport_var13_1y3[full$delta_imp_aport_var13_1y3==9999999999]=10


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

##############################

#### FEATURE ENGINEERING #####

##############################

##### 0 count per line
count0 <- function(x) {  return( sum(x == 0 & !is.na(x)) ) }
count3 <- function(x) {  return( sum(x == 3 & !is.na(x)) ) }
count6 <- function(x) {  return( sum(x == 6 & !is.na(x)) ) }
count9 <- function(x) {  return( sum(x == 9 & !is.na(x)) ) }
count_mod3 <- function(x) {  return( sum(x > 0 & round(x,2) %% 3 == 0 & !is.na(x)) ) }

full$n0 <- apply(full[,!names(full) %in% c('ID','TARGET','set1','set2','set3','set4','set5','set6','set7','set8','set9','set10','set11','set12','set13','set14','set15','set16','set17','set18','set19','set20','var3','var15','var36','var38','num_var4','s0','s1','s2','s3')], 1, FUN=count0)
full$n3 <- apply(full[,!names(full) %in% c('ID','TARGET','set1','set2','set3','set4','set5','set6','set7','set8','set9','set10','set11','set12','set13','set14','set15','set16','set17','set18','set19','set20','var3','var15','var36','var38','num_var4','s0','s1','s2','s3','n0')], 1, FUN=count3)
full$n6 <- apply(full[,!names(full) %in% c('ID','TARGET','set1','set2','set3','set4','set5','set6','set7','set8','set9','set10','set11','set12','set13','set14','set15','set16','set17','set18','set19','set20','var3','var15','var36','var38','num_var4','s0','s1','s2','s3','n0','n3')], 1, FUN=count6)
full$n9 <- apply(full[,!names(full) %in% c('ID','TARGET','set1','set2','set3','set4','set5','set6','set7','set8','set9','set10','set11','set12','set13','set14','set15','set16','set17','set18','set19','set20','var3','var15','var36','var38','num_var4','s0','s1','s2','s3','n0','n3','n6')], 1, FUN=count9)
full$n_mod3 <- apply(full[,!names(full) %in% c('ID','TARGET','set1','set2','set3','set4','set5','set6','set7','set8','set9','set10','set11','set12','set13','set14','set15','set16','set17','set18','set19','set20','var3','var15','var36','var38','num_var4','s0','s1','s2','s3','n0','n3','n6','n9')], 1, FUN=count_mod3)


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



#######################
###ult1/ult3 ratios####
#######################

full$imp_op_var39_comer_ult_ratio=full$imp_op_var39_comer_ult1/full$imp_op_var39_comer_ult3
full$imp_op_var41_comer_ult_ratio=full$imp_op_var41_comer_ult1/full$imp_op_var41_comer_ult3
full$imp_op_var39_efect_ult1=full$imp_op_var39_efect_ult1/full$imp_op_var39_efect_ult3
full$imp_op_var41_efect_ult1=full$imp_op_var41_efect_ult1/full$imp_op_var41_efect_ult3
full$num_op_var39_comer_ult_ratio=full$num_op_var39_comer_ult1/full$num_op_var39_comer_ult3
full$num_op_var41_comer_ult_ratio=full$num_op_var41_comer_ult1/full$num_op_var41_comer_ult3
full$num_op_var39_efect_ult1=full$num_op_var39_efect_ult1/full$num_op_var39_efect_ult3
full$num_op_var41_efect_ult1=full$num_op_var41_efect_ult1/full$num_op_var41_efect_ult3
full$num_op_var39_ult_ratio=full$num_op_var39_ult1/full$num_op_var39_ult3
full$num_op_var41_ult_ratio=full$num_op_var41_ult1/full$num_op_var41_ult3
full$num_var22_ult_ratio=full$num_var22_ult1/full$num_var22_ult3
full$num_var45_ult_ratio=full$num_var45_ult1/full$num_var45_ult3
full$saldo_medio_var5_ult_ratio=full$saldo_medio_var5_ult1/full$saldo_medio_var5_ult3
full$num_var22_hace_ratio=full$num_var22_hace2/full$num_var22_hace3
full$num_var45_hace_ratio=full$num_var45_hace2/full$num_var45_hace3
full$saldo_medio_var5_hace_ratio=full$saldo_medio_var5_hace2/full$saldo_medio_var5_hace3
full[full==Inf]=-99999
full[full==-Inf]=-99999



#####################
#### X modulus 3 ####
#####################

full$imp_ent_var16_ult1r=ifelse(round(full$imp_ent_var16_ult1,2)%%3==0,-1,1);full$imp_ent_var16_ult1r[full$imp_ent_var16_ult1==0]=0
full$imp_op_var39_comer_ult1r=ifelse(round(full$imp_op_var39_comer_ult1,2)%%3==0,-1,1);full$imp_op_var39_comer_ult1r[full$imp_op_var39_comer_ult1==0]=0
full$imp_op_var39_comer_ult3r=ifelse(round(full$imp_op_var39_comer_ult3,2)%%3==0,-1,1);full$imp_op_var39_comer_ult3r[full$imp_op_var39_comer_ult3==0]=0
full$imp_op_var41_comer_ult1r=ifelse(round(full$imp_op_var41_comer_ult1,2)%%3==0,-1,1);full$imp_op_var41_comer_ult1r[full$imp_op_var41_comer_ult1==0]=0
full$imp_op_var41_comer_ult3r=ifelse(round(full$imp_op_var41_comer_ult3,2)%%3==0,-1,1);full$imp_op_var41_comer_ult3r[full$imp_op_var41_comer_ult3==0]=0
full$imp_op_var41_efect_ult1r=ifelse(round(full$imp_op_var41_efect_ult1,2)%%3==0,-1,1);full$imp_op_var41_efect_ult1r[full$imp_op_var41_efect_ult1==0]=0
full$imp_op_var41_efect_ult3r=ifelse(round(full$imp_op_var41_efect_ult3,2)%%3==0,-1,1);full$imp_op_var41_efect_ult3r[full$imp_op_var41_efect_ult3==0]=0
full$imp_op_var41_ult1r=ifelse(round(full$imp_op_var41_ult1,2)%%3==0,-1,1);full$imp_op_var41_ult1r[full$imp_op_var41_ult1==0]=0
full$imp_op_var39_efect_ult1r=ifelse(round(full$imp_op_var39_efect_ult1,2)%%3==0,-1,1);full$imp_op_var39_efect_ult1r[full$imp_op_var39_efect_ult1==0]=0
full$imp_op_var39_efect_ult3r=ifelse(round(full$imp_op_var39_efect_ult3,2)%%3==0,-1,1);full$imp_op_var39_efect_ult3r[full$imp_op_var39_efect_ult3==0]=0
full$imp_op_var39_ult1r=ifelse(round(full$imp_op_var39_ult1,2)%%3==0,-1,1);full$imp_op_var39_ult1r[full$imp_op_var39_ult1==0]=0
full$saldo_var5r=ifelse(round(full$saldo_var5,2)%%3==0,-1,1);full$saldo_var5r[full$saldo_var5==0]=0
full$saldo_var30r=ifelse(round(full$saldo_var30,2)%%3==0,-1,1);full$saldo_var30r[full$saldo_var30==0]=0
full$saldo_var42r=ifelse(round(full$saldo_var42,2)%%3==0,-1,1);full$saldo_var42r[full$saldo_var42==0]=0
full$imp_var43_emit_ult1r=ifelse(round(full$imp_var43_emit_ult1,2)%%3==0,-1,1);full$imp_var43_emit_ult1r[full$imp_var43_emit_ult1==0]=0
full$saldo_medio_var5_hace2r=ifelse(round(full$saldo_medio_var5_hace2,2)%%3==0,-1,1);full$saldo_medio_var5_hace2r[full$saldo_medio_var5_hace2==0]=0                  
full$saldo_medio_var5_hace3r=ifelse(round(full$saldo_medio_var5_hace3,2)%%3==0,-1,1);full$saldo_medio_var5_hace3r[full$saldo_medio_var5_hace3==0]=0  
full$saldo_medio_var5_ult1r=ifelse(round(full$saldo_medio_var5_ult1,2)%%3==0,-1,1);full$saldo_medio_var5_ult1r[full$saldo_medio_var5_ult1==0]=0  
full$saldo_medio_var5_ult3r=ifelse(round(full$saldo_medio_var5_ult3,2)%%3==0,-1,1);full$saldo_medio_var5_ult3r[full$saldo_medio_var5_ult3==0]=0  


feature.names=setdiff(names(full),c('ID','TARGET',
                                    'set1','set2','set3','set4','set5',
                                    'set6','set7','set8','set9','set10',
                                    'set11','set12','set13','set14','set15',
                                    'set16','set17','set18','set19','set20'
))


####xgboost params
param1 <- list(
  objective  = "binary:logistic"
  , eval_metric = "auc"
  , eta = 0.03
  , subsample = 0.7
  , colsample_bytree = 0.6
  , min_child_weight = 1
  , max_depth = 5
)


dtrain1=list();dtrain2=list();dtrain3=list();dtrain4=list();dtrain5=list()
dval1=list();dval2=list();dval3=list();dval4=list();dval5=list()

for (i in 1:20) {
  set=full[[names(full)[grep('set',names(full))][i]]]
  dtrain1[[i]] <- xgb.DMatrix(data.matrix(full[!is.na(full$TARGET) & set!=0,feature.names]), label=full$TARGET[!is.na(full$TARGET) & set!=0],missing=NaN)
  dtrain2[[i]] <- xgb.DMatrix(data.matrix(full[!is.na(full$TARGET) & set!=1,feature.names]), label=full$TARGET[!is.na(full$TARGET) & set!=1],missing=NaN)
  dtrain3[[i]] <- xgb.DMatrix(data.matrix(full[!is.na(full$TARGET) & set!=2,feature.names]), label=full$TARGET[!is.na(full$TARGET) & set!=2],missing=NaN)
  dtrain4[[i]] <- xgb.DMatrix(data.matrix(full[!is.na(full$TARGET) & set!=3,feature.names]), label=full$TARGET[!is.na(full$TARGET) & set!=3],missing=NaN)
  dtrain5[[i]] <- xgb.DMatrix(data.matrix(full[!is.na(full$TARGET) & set!=4,feature.names]), label=full$TARGET[!is.na(full$TARGET) & set!=4],missing=NaN)
  dval1[[i]] <- xgb.DMatrix(data.matrix(full[!is.na(full$TARGET) & set==0,feature.names]), label=full$TARGET[!is.na(full$TARGET) & set==0],missing=NaN)
  dval2[[i]] <- xgb.DMatrix(data.matrix(full[!is.na(full$TARGET) & set==1,feature.names]), label=full$TARGET[!is.na(full$TARGET) & set==1],missing=NaN)
  dval3[[i]] <- xgb.DMatrix(data.matrix(full[!is.na(full$TARGET) & set==2,feature.names]), label=full$TARGET[!is.na(full$TARGET) & set==2],missing=NaN)
  dval4[[i]] <- xgb.DMatrix(data.matrix(full[!is.na(full$TARGET) & set==3,feature.names]), label=full$TARGET[!is.na(full$TARGET) & set==3],missing=NaN)
  dval5[[i]] <- xgb.DMatrix(data.matrix(full[!is.na(full$TARGET) & set==4,feature.names]), label=full$TARGET[!is.na(full$TARGET) & set==4],missing=NaN)  
}
dtest <- xgb.DMatrix(data.matrix(full[is.na(full$TARGET),feature.names]), label=full$TARGET[is.na(full$TARGET)],missing=NaN)

xgb.model1=list();xgb.model2=list();xgb.model3=list();xgb.model4=list();xgb.model5=list()
for (i in 1:20) {
  print(paste(i,1)); set.seed(i*1); xgb.model1[[i]]<- xgb.train(params=param1,data=dtrain1[[i]],nrounds=270,verbose=0,nthread=32,maximize=TRUE)
  print(paste(i,2)); set.seed(i*2); xgb.model2[[i]]<- xgb.train(params=param1,data=dtrain2[[i]],nrounds=270,verbose=0,nthread=32,maximize=TRUE)
  print(paste(i,3)); set.seed(i*3); xgb.model3[[i]]<- xgb.train(params=param1,data=dtrain3[[i]],nrounds=270,verbose=0,nthread=32,maximize=TRUE)
  print(paste(i,4)); set.seed(i*4); xgb.model4[[i]]<- xgb.train(params=param1,data=dtrain4[[i]],nrounds=270,verbose=0,nthread=32,maximize=TRUE)
  print(paste(i,5)); set.seed(i*5); xgb.model5[[i]]<- xgb.train(params=param1,data=dtrain5[[i]],nrounds=270,verbose=0,nthread=32,maximize=TRUE)
}


cv.set=list()
sets=c('set1','set2','set3','set4','set5','set6','set7','set8','set9','set10','set11','set12','set13','set14','set15','set16','set17','set18','set19','set20')

for (i in 1:20) {
  print(i)
  cv.set[[i]]=rep(NA,dim(full)[1])
  set=full[[sets[i]]]
  cv.set[[i]][set==0 & !is.na(set)]=predict(xgb.model1[[i]],dval1[[i]])
  cv.set[[i]][set==1 & !is.na(set)]=predict(xgb.model2[[i]],dval2[[i]])
  cv.set[[i]][set==2 & !is.na(set)]=predict(xgb.model3[[i]],dval3[[i]])
  cv.set[[i]][set==3 & !is.na(set)]=predict(xgb.model4[[i]],dval4[[i]])
  cv.set[[i]][set==4 & !is.na(set)]=predict(xgb.model5[[i]],dval5[[i]])
  cv.set[[i]][is.na(set)]=
    (predict(xgb.model1[[i]],dtest)+
       predict(xgb.model2[[i]],dtest)+
       predict(xgb.model3[[i]],dtest)+
       predict(xgb.model4[[i]],dtest)+
       predict(xgb.model5[[i]],dtest)
    )/5
  cv.set[[i]]=rank(cv.set[[i]])
}

full$xgb_raddar2=Reduce('+',cv.set)/20

write.csv(select(filter(full,!is.na(TARGET)),ID,xgb_raddar2),'data/output/temp/xgb_raddar_2_train.csv',row.names=FALSE)
write.csv(select(filter(full,is.na(TARGET)),ID,xgb_raddar2),'data/output/temp/xgb_raddar_2_test.csv',row.names=FALSE)












