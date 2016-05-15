import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import roc_auc_score


def loadcolumn(filename,col=4, skip=1, floats=True):
    pred=[]
    op=open(filename,'r')
    if skip==1:
        op.readline() #header
    for line in op:
        line=line.replace('\n','')
        sps=line.split(',')
        #load always the last columns
        if floats:
            pred.append(float(sps[col]))
        else :
            pred.append(str(sps[col]))
    op.close()
    return pred            

    
def load_datas(filename):

    return joblib.load(filename)

def printfile(X, filename):

    joblib.dump((X), filename)
    
def printfilcsve(X, filename, headers):

    np.savetxt(filename,X, header=headers) 

    
def load_ids(id_file):

    idss=loadcolumn(id_file,col=1, skip=1, floats=True)
    id_list=[ [] ,[] , [], [] , []]
    for g in range(0,len(idss)):
        id_list[int(idss[g])].append(g)
    biglist=[]
    for k in range(5):
        training_ids=[s for s in range(0,len(idss)) if s not in id_list[k] ]
        biglist.append([training_ids,id_list[k] ])
        print(len(biglist), len(biglist[0]))
    return biglist
    



    
def main():
    
       
        metafolder_train="data/output/train/"
        metafolder_test="data/output/test/"    
        input_folder="data/input/"        

        ######### Load files ############
        y=loadcolumn(input_folder+ "train.csv",col=370, skip=1, floats=True)

        #variables to load
        meta=["xgboost_marios_1","xgboost_marios_2","xgboost_marios_4","xgboost_marios_5","rf_marios_1", "nn_marios_1"
                ,"nn_marios_1_2","et_marios_1_2","rf_marios_1_2", "nn_marios_1_3","xgboost_marios_1_2","knn_marios_1","ada_marios_7_2","ada_marios_7_3","xgboost_marios_7_2" ]

        
        print("len of target=%d" % (len(y))) # reconciliation check

        Xmetatrain=None
        Xmetatest=None   
        #append all the predictions into 1 list (array)
        for modelname in meta :
            mini_xtrain=np.loadtxt(metafolder_train+ modelname + '.train.csv', skiprows=1, usecols=[1])
            mini_xtest=np.loadtxt(metafolder_test + modelname + '.test.csv', skiprows=1, usecols=[1])  
            mean_train=np.mean(mini_xtrain)
            mean_test=np.mean(mini_xtest)               
            print("model %s auc %f mean train/test %f/%f " % (modelname,roc_auc_score(y,mini_xtrain) ,mean_train,mean_test)) 
            if Xmetatrain==None:
                Xmetatrain=mini_xtrain
                Xmetatest=mini_xtest
            else :
                Xmetatrain=np.column_stack((Xmetatrain,mini_xtrain))
                Xmetatest=np.column_stack((Xmetatest,mini_xtest))
        # convert my scores to list

        X=Xmetatrain
        X_test=Xmetatest


        np.savetxt(metafolder_train+ "marios_models_V" + str(len(meta)) + ".train.csv", X , header=" ".join([k for k in meta])) 
        np.savetxt(metafolder_test+ "marios_models_V" + str(len(meta)) + ".test.csv", X_test , header=" ".join([k for k in meta]))
         

       
       
       



if __name__=="__main__":
  main()
