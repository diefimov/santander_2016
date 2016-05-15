library(dplyr)

##### Read data
train <- read.csv("data/input/train.csv")
test <- read.csv("data/input/test.csv")
test$TARGET=NA

full=rbind(train,test)

full$var38[full$var38==117310.979016494]=NA
full$var15[full$var15<=22]=22
full$var15[full$var15>=95]=95


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

full$var38[full$var38==117310.979016494]=NA
full$zeroSegment=apply(full[,setdiff(names(full),c('ID','TARGET','idx',
                                                   'var3','var15','var36','var38',
                                                   'num_var4',
                                                   'num_var1_0',
                                                   'num_var2_0',
                                                   'num_var5_0',
                                                   'num_var6_0',
                                                   'num_var8_0',
                                                   'num_var13_corto_0',
                                                   'num_var13_medio_0',        
                                                   'num_var13_largo_0',                                    
                                                   'num_var14_0',
                                                   'num_var18_0',
                                                   'num_var20_0',
                                                   'num_var22_hace2',
                                                   'num_var22_hace3',
                                                   'num_med_var22_ult3',
                                                   'num_var22_ult1',
                                                   'num_var22_ult3',
                                                   'num_var24_0',
                                                   'num_var25_0',
                                                   'num_var26_0',
                                                   'num_var27_0',
                                                   'num_var28_0',
                                                   'num_var29_0',
                                                   'num_var30_0',
                                                   'num_var32_0',
                                                   'num_var35_0',
                                                   'num_var37_0',
                                                   'num_var39_0',
                                                   'num_meses_var39_vig_ult3',
                                                   'num_var40_0',
                                                   'num_var41_0',
                                                   'num_var42_0',
                                                   'num_var44_0',
                                                   'num_var45_hace2',
                                                   'num_var45_hace3',
                                                   'num_med_var45_ult3',
                                                   'num_var45_ult1',
                                                   'num_var45_ult3',
                                                   'num_var46_0'))],1,sum)


full$zeroSegment[full$zeroSegment!=0]=1
full$zeroSegment[full$var36!=99 & full$zeroSegment==0]=1
full$zeroSegment[full$var15<=22 & full$zeroSegment==0]=1
full$zeroSegment[full$var3!=2 & full$zeroSegment==0]=1
full$zeroSegment[full$num_var22_ult1!=0 & full$zeroSegment==0]=1
full$zeroSegment[full$num_meses_var39_vig_ult3==1 & full$zeroSegment==0]=1

write.csv(select(full,ID,zeroSegment),'data/output/features/population_split.csv',row.names=F)





