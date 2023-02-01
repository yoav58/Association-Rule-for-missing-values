import pandas as pd
from sklearn.model_selection import train_test_split
from AprioriForNull import AprioriForNull
from  MachineLearningModels import  MachineLearningModels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from NullValuesFiller import NullValuesFiller

import Helper


def association_rule_method(confident):
    dtf = pd.read_csv('Titanic-Dataset.csv')

    # drop non relevant columns
    non_rel_col = ['Name','Ticket','PassengerId','Cabin']
    dtf = dtf.drop(non_rel_col,axis=1)


    # remove outliers
    dtf = Helper.HelperMethods.remove_outliers(dtf)

    # split to train and test
    dtf_train, dtf_test = train_test_split(dtf,test_size=0.25)


    # convert to categorical
    categorical_train = dtf_train.copy().drop('Survived',axis=1)
    categorical_test = dtf_test.copy().drop('Survived',axis=1)

    # categories like Pclass,SibSp and Parch can directly be converted since their range of possible values
    # is quite low.
    cat_att = ['Pclass','SibSp','Parch']
    Helper.HelperMethods.convert_with_astype(categorical_train,cat_att)
    Helper.HelperMethods.convert_with_astype(categorical_test,cat_att)

    # convert age
    age_lable = ['young','Adults','olds']
    age_range = [0, 21, 45, 100]
    age_directory = Helper.HelperMethods.rangeMeanDictionary(categorical_train,age_range,age_lable,'Age')
    categorical_train['Age'] = Helper.HelperMethods.convertToCategorial(categorical_train,age_range,age_lable,'Age')
    categorical_test['Age'] = Helper.HelperMethods.convertToCategorial(categorical_test,age_range,age_lable,'Age')
    # convert Fare
    fare_lable = ['low','med','high']
    fare_range = [-1,52,102,513]
    fare_directory = Helper.HelperMethods.rangeMeanDictionary(categorical_train,fare_range,fare_lable,'Fare')
    categorical_train['Fare'] = Helper.HelperMethods.convertToCategorial(categorical_train,fare_range,fare_lable,'Fare')
    categorical_test['Fare'] = Helper.HelperMethods.convertToCategorial(categorical_test,fare_range,fare_lable,'Fare')
    print(categorical_train.dtypes)

    # dictionary to be able restore data
    converted_train_dictionary = {
        "Age": age_directory,
        "Fare":fare_directory,
    }
    typeDictionary = dtf_train.dtypes.to_dict()
    # find rules
    print(dtf_train.isna().sum())
    apriori_For_null = AprioriForNull
    itemsets, rules =  apriori_For_null.findRules(categorical_train, 0.3, confident)
    apriori_For_null.fill_nulls_with_apriori(self=apriori_For_null,train_data=dtf_train,categorical_data=categorical_train,rules=rules,convertedDictionary=converted_train_dictionary,typeDictionary=typeDictionary)
    apriori_For_null.fill_nulls_with_apriori(self=apriori_For_null,train_data=dtf_test,categorical_data=categorical_test,rules=rules,convertedDictionary=converted_train_dictionary,typeDictionary=typeDictionary)
    NullValuesFiller.mean(dtf_train, dtf_train, ['Age'])
    NullValuesFiller.mean(dtf_test, dtf_train, ['Age'])
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
    return acc;