import pandas as pd
from sklearn.model_selection import train_test_split
from NullValuesFiller import NullValuesFiller
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score






##################################################################################################
# in this file i used machine learning to predict the OUTCOME, i used simple approach to fill the
# null values (mean) also, since OUTCOME is binary attribute, i used logistic regression model
# which should be more appropriate
##################################################################################################
import Helper

def simpleMethod():
    dtf = pd.read_csv('Car_Insurance_Claim.csv').drop('ID', axis=1)

    # drop columns that are not relevant or could effect the performance too much
    not_rel = ['POSTAL_CODE','DUIS']
    dtf = dtf.drop(not_rel,axis=1)

    # remove outliers
    dtf = Helper.HelperMethods.remove_outliers(dtf)

    # split to train,test
    dtf_train, dtf_test = train_test_split(dtf, test_size=0.25)

    # see the amount of null values
    print(dtf_train.isna().sum())

    # fill the null values with the mean
    f_at = ['CREDIT_SCORE','ANNUAL_MILEAGE']
    dtf_train = NullValuesFiller.mean(dtf_train,dtf_train,f_at)
    dtf_test = NullValuesFiller.mean(dtf_test,dtf_train,f_at) # used the mean of the train to prevent data leakage.
    print(dtf_train.isna().sum())
    print(dtf_test.isna().sum())

    # convert to numerical ('One Hot Encoding')
    dtf_train = pd.get_dummies(dtf_train)
    dtf_test = pd.get_dummies(dtf_test)

    # start machine learning prediction
    X_train = dtf_train.copy().drop('OUTCOME',axis=1)
    X_test = dtf_test.copy().drop('OUTCOME',axis=1)
    Y_train = dtf_train["OUTCOME"]
    Y_test = dtf_test['OUTCOME']

    # using logistic regression model since the attribute OUTCOME is binary
    clf = LogisticRegression(max_iter=10000)
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



