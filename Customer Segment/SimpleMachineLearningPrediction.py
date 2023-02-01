import pandas as pd
from sklearn.model_selection import train_test_split
from NullValuesFiller import NullValuesFiller
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import Helper
from  MachineLearningModels import  MachineLearningModels




'''""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
in this file i used machine learning to predict the Spending_Score, i used simple approach to fill the
null values (mean) and for categorical attributes i filled with the must frequent value.
'''""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def simple_method():
    # read the files (provided splatted)
    dtf_train = pd.read_csv('Train.csv').drop(['ID','Segmentation'], axis=1)
    dtf_test = pd.read_csv('Test.csv').drop('ID',axis=1)

    # fill the null values

    # with mean
    numeric_null_attributes = ['Work_Experience','Family_Size']
    NullValuesFiller.mean(dtf_train,dtf_train,numeric_null_attributes)
    NullValuesFiller.mean(dtf_test,dtf_train,numeric_null_attributes) # use train must frequant to prevent leakage.

    # fill with must frequant
    categorical_null_attributes = ['Ever_Married', 'Graduated', 'Profession', 'Var_1']
    NullValuesFiller.mustFrequent(dtf_train,categorical_null_attributes)
    NullValuesFiller.mustFrequentTest(dtf_test,dtf_train,categorical_null_attributes)

    # before "one hot encoding" convert the spending score to numeric,
    cat_values = ['Low','Average','High']
    num_values = [1,2,3]
    dtf_train['Spending_Score'].replace(cat_values,num_values , inplace=True)
    Helper.HelperMethods.convert_to_num(dtf_train,['Spending_Score'])
    dtf_test['Spending_Score'].replace(cat_values,num_values, inplace=True)
    Helper.HelperMethods.convert_to_num(dtf_test,['Spending_Score'])

    # use "one hot encoding"
    dtf_train = pd.get_dummies(dtf_train)
    dtf_test = pd.get_dummies(dtf_test)

    # convert spending score back to categorical
    dtf_train['Spending_Score'].replace(num_values, cat_values, inplace=True)
    Helper.HelperMethods.convert_with_astype(dtf_train,['Spending_Score'])
    dtf_test['Spending_Score'].replace(num_values,cat_values, inplace=True)
    Helper.HelperMethods.convert_with_astype(dtf_test,['Spending_Score'])




    prediction, Y_test = MachineLearningModels.LogisticRegressionModel(dtf_train,dtf_test,'Spending_Score')
    acc = accuracy_score(Y_test, prediction)
    prec = precision_score(Y_test, prediction, average='micro')
    rec = recall_score(Y_test, prediction,average='micro')
    f1 = f1_score(Y_test, prediction,average='micro')
    print("the accuracy is: ", acc)
    print("the precision is: ", prec)
    print("the recall score is: ", rec)
    print("the f1 score is: ", f1)
    return acc










