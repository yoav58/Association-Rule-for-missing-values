import pandas as pd
from AprioriForNull import AprioriForNull
from NullValuesFiller import NullValuesFiller
from  MachineLearningModels import  MachineLearningModels
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score





# read the files (provided splatted)
import Helper

def association_rule_method(confident):
    dtf_train = pd.read_csv('Train.csv').drop(['ID','Segmentation'], axis=1)
    dtf_test = pd.read_csv('Test.csv').drop('ID',axis=1)



    """
    start data pre proccesing for the apriori algorithm, First, convert the numeric attributes
    to categorical.
    """
    categorical_train = dtf_train.copy().drop('Spending_Score',axis=1)
    categorical_test = dtf_test.copy().drop('Spending_Score',axis=1)

    print(dtf_train.isna().sum())

    # convert Age
    age_label = ['Youth ','Adults','middleAged','Seniors']
    age_range = [0, 24, 31, 55, 100]
    age_directory = Helper.HelperMethods.rangeMeanDictionary(categorical_train,age_range,age_label,'Age')
    categorical_train['Age'] = Helper.HelperMethods.convertToCategorial(categorical_train,age_range,age_label,'Age')
    categorical_test['Age'] = Helper.HelperMethods.convertToCategorial(categorical_test,age_range,age_label,'Age')

    # convert Work_Experience
    work_label = ['low','med','high']
    work_range = [-1,1,2,10]
    work_directory = Helper.HelperMethods.rangeMeanDictionary(categorical_train,age_range,age_label,'Work_Experience')
    categorical_train['Work_Experience'] = Helper.HelperMethods.convertToCategorial(categorical_train,age_range,age_label,'Work_Experience')
    categorical_test['Work_Experience'] = Helper.HelperMethods.convertToCategorial(categorical_test,age_range,age_label,'Work_Experience')

    # convert Family_Size
    work_label = ['low','med','high']
    work_range = [-1,2,4,10]
    family_directory = Helper.HelperMethods.rangeMeanDictionary(categorical_train,work_range,work_label,'Family_Size')
    categorical_train['Family_Size'] = Helper.HelperMethods.convertToCategorial(categorical_train,work_range,work_label,'Family_Size')
    categorical_test['Family_Size'] = Helper.HelperMethods.convertToCategorial(categorical_test,work_range,work_label,'Family_Size')

    # dictionary to be able restore data
    converted_train_dictionary = {
        "Age": age_directory,
        "Family_Size":family_directory,
        "Work_Experience": work_directory
    }
    typeDictionary = dtf_train.dtypes.to_dict()



    # since in this data set some rules contains nan, delete those rows from the categorical train.
    categorical_train = categorical_train.dropna()

    # find rules
    apriori_For_null = AprioriForNull
    itemsets, rules =  apriori_For_null.findRules(categorical_train, 0.3, confident)

    apriori_For_null.fill_nulls_with_apriori(self=apriori_For_null,train_data=dtf_train,categorical_data=categorical_train,rules=rules,convertedDictionary=converted_train_dictionary,typeDictionary=typeDictionary)
    apriori_For_null.fill_nulls_with_apriori(self=apriori_For_null,train_data=dtf_test,categorical_data=categorical_test,rules=rules,convertedDictionary=converted_train_dictionary,typeDictionary=typeDictionary)
    print(dtf_train.isna().sum())


    # fill the remaning values
    c_at = ['Ever_Married','Graduated','Profession','Var_1']
    NullValuesFiller.mustFrequent(dtf_train,c_at)
    NullValuesFiller.mustFrequentTest(dtf_test,dtf_train,c_at)
    n_att = ['Work_Experience','Family_Size']
    dtf_train = NullValuesFiller.mean(dtf_train,dtf_train,n_att)
    dtf_test = NullValuesFiller.mean(dtf_test,dtf_train,n_att)

    print(dtf_train.isna().sum())



    """
    machine learning step
    """
    # before "one hot encoding" convert the spending score to numeric,
    cat_values = ['Low','Average','High']
    num_values = [1,2,3]
    dtf_train['Spending_Score'].replace(cat_values,num_values , inplace=True)
    Helper.HelperMethods.convert_to_num(dtf_train,['Spending_Score'])
    dtf_test['Spending_Score'].replace(cat_values,num_values, inplace=True)
    Helper.HelperMethods.convert_to_num(dtf_test,['Spending_Score'])

    dtf_train = pd.get_dummies(dtf_train)
    dtf_test = pd.get_dummies(dtf_test)


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
