import pandas as pd
from sklearn.model_selection import train_test_split
from NullValuesFiller import NullValuesFiller
from  MachineLearningModels import  MachineLearningModels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


import Helper

'''""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
in this file i used machine learning to predict the Survived attribute, in Titanic ship.
i used simple method (mean/must frequent) to fill the null values.
'''""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def simple_method():
    dtf = pd.read_csv('Titanic-Dataset.csv')

    # drop columns that are not relevant or could effet the performance too much (like name)
    non_rel_col = ['Name','Ticket','PassengerId','Cabin']
    dtf = dtf.drop(non_rel_col,axis=1)

    # remove outliers
    dtf = Helper.HelperMethods.remove_outliers(dtf)



    ##################################################
    # pre-processing the the test and train
    ##################################################
    dtf_train, dtf_test = train_test_split(dtf,test_size=0.25)

    # fill the null values
    NullValuesFiller.mean(dtf_train,dtf_train,['Age'])
    NullValuesFiller.mean(dtf_test,dtf_train,['Age'])

    # convert to numeric
    dtf_train = pd.get_dummies(dtf_train)
    dtf_test = pd.get_dummies(dtf_test)

    prediction, Y_test = MachineLearningModels.LogisticRegressionModel(dtf_train,dtf_test,'Survived')
    acc = accuracy_score(Y_test, prediction)
    prec = precision_score(Y_test, prediction, average='micro')
    rec = recall_score(Y_test, prediction,average='micro')
    f1 = f1_score(Y_test, prediction,average='micro')
    print("the accuracy is: ", acc)
    print("the precision is: ", prec)
    print("the recall score is: ", rec)
    print("the f1 score is: ", f1)
    return acc


