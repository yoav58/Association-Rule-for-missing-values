import pandas as pd
from  Helper import HelperMethods
from sklearn.model_selection import train_test_split
from AprioriForNull import AprioriForNull
from NullValuesFiller import NullValuesFiller
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score









##################################################################################################
# in this file i used association rules to fill the null values.
##################################################################################################
def association_rule_method(codfident):
    # read the file
    dtf = pd.read_csv('Car_Insurance_Claim.csv').drop('ID', axis=1)



    # drop columns that arent relevant or could effect the performance too much.
    not_rel = ['POSTAL_CODE','DUIS','VEHICLE_TYPE']
    dtf = dtf.drop(not_rel,axis=1)

    # remove outliers
    dtf = HelperMethods.remove_outliers(dtf)

    # some of the numrical attribute can be "look" like catogircal because they have low range of values. so convert
    # those attributes
    list_to_convert = ['VEHICLE_OWNERSHIP','MARRIED','CHILDREN','SPEEDING_VIOLATIONS','PAST_ACCIDENTS']
    HelperMethods.convert_with_astype(dtf,list_to_convert)


    # split the data
    dtf_train, dtf_test = dtf_train, dtf_test = train_test_split(dtf,test_size=0.25)
    train_outcome = dtf_train['OUTCOME']
    test_outcome = dtf_test['OUTCOME']

    # convert to categorical (using 4 bins)
    categorical_train = dtf_train.copy().drop('OUTCOME',axis=1)
    categorical_test = dtf_test.copy().drop('OUTCOME',axis=1)

    #################################################
    # convert to categorical.
    ##################################################

    # convert  credit score, i save also dictionary of the mean of each lable so after using apriori
    # i can convert back to the mean of the bin.
    credit_score_label = ['low','medium','high','veryHigh']
    categorical_train, credit_train_dict = HelperMethods.equal_convertToCategorical(categorical_train,credit_score_label,4,'CREDIT_SCORE')
    categorical_test, credit_test_dict = HelperMethods.equal_convertToCategorical(categorical_test,credit_score_label,4,'CREDIT_SCORE')

    # convert ANNUAL_MILEAGE the same way converted creditScore
    ANNUAL_MILEAGE_label = ['low','medium','high','veryHigh']
    categorical_train, annual_train_dict = HelperMethods.equal_convertToCategorical(categorical_train,ANNUAL_MILEAGE_label,4,'ANNUAL_MILEAGE')
    categorical_test, annual_test_dict = HelperMethods.equal_convertToCategorical(categorical_test,ANNUAL_MILEAGE_label,4,'ANNUAL_MILEAGE')

    converted_train_dictionary = {
        "CREDIT_SCORE": credit_train_dict,
        "ANNUAL_MILEAGE": annual_train_dict,
    }

    typeDictionary = dtf_train.dtypes.to_dict()

    #################################################
    # fill null values.
    ##################################################

    # find the rules
    print(dtf_train.isna().sum())
    print(categorical_train.dtypes)
    apriori_For_null = AprioriForNull
    itemsets, rules = apriori_For_null.findRules(categorical_train, 0.3, 0.6)
    # fill the values in train and test
    apriori_For_null.fill_nulls_with_apriori(self=apriori_For_null,train_data=dtf_train,categorical_data=categorical_train,rules=rules,convertedDictionary=converted_train_dictionary,typeDictionary=typeDictionary)
    apriori_For_null.fill_nulls_with_apriori(self=apriori_For_null,train_data=dtf_test,categorical_data=categorical_test,rules=rules,convertedDictionary=converted_train_dictionary,typeDictionary=typeDictionary)

    # fill the remaining values with the mean
    f_at = ['CREDIT_SCORE','ANNUAL_MILEAGE']
    dtf_train = NullValuesFiller.mean(dtf_train,dtf_train,f_at)
    dtf_test = NullValuesFiller.mean(dtf_test,dtf_train,f_at)


    #################################################
    # start ml prediction
    ##################################################

    # convert the attributes back to numeric, first convert back with astype the attributes who converted before,
    # then convert credit score and annual mileage.
    HelperMethods.convert_to_num(dtf_train,list_to_convert)
    HelperMethods.convert_to_num(dtf_test,list_to_convert)
    dtf_train = pd.get_dummies(dtf_train)
    dtf_test = pd.get_dummies(dtf_test)





    # start machine learning prediction
    X_train = dtf_train.copy().drop('OUTCOME',axis=1)
    X_test = dtf_test.copy().drop('OUTCOME',axis=1)
    Y_train = dtf_train["OUTCOME"]
    Y_test = dtf_test['OUTCOME']

    # using logistic regression model since the attribute OUTCOME is binary
    clf = LogisticRegression(max_iter=100000)
    clf.fit(X_train,Y_train)
    prediction = clf.predict(X_test)
    acc = accuracy_score(Y_test, prediction)
    prec = precision_score(Y_test, prediction)
    rec = recall_score(Y_test, prediction)
    f1 = f1_score(Y_test, prediction)
    print("the accuracy is: ",acc)
    print("the precision is: ",prec)
    print("the recall score is: ",rec)
    print("the f1 score is: ",f1)
    return acc
