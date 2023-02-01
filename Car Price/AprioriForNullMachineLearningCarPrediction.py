import pandas as pd
from NullValuesFiller import NullValuesFiller
from Helper import HelperMethods
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from AprioriForNull import AprioriForNull
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_percentage_error,mean_absolute_error
import numpy as np # linear algebra


def association_rule_method(confident):
    # read file csv
    dtf = pd.read_csv("../autos.csv").drop('index', axis=1)

    # drop columns that are not relevant or could effect the performance too much
    cols = ["nrOfPictures", "lastSeen", "dateCrawled", "name", "monthOfRegistration", "dateCreated", "postalCode", "seller", "offerType"]
    car_data = dtf.drop(cols, axis=1)
    # remove outliers by using IQR
    car_data = HelperMethods.remove_outliers(car_data)

    ##################################################
    # pre-processing the the test and train
    ##################################################
    dtf_train, dtf_test = train_test_split(car_data,test_size=0.25)
    train_price = dtf_train['price']
    test_price = dtf_test['price']
    #dtf_train = dtf_train.drop(['price'],axis=1)
    #dtf_test = dtf_test.drop(['price'],axis=1)

    # now fill the null values, first, since model can effect the performance too much, replace him with the must common
    NullValuesFiller.mustFrequent(dtf_train,['model'])
    NullValuesFiller.mustFrequent(dtf_test,['model'])

    # now convert to categorical to be able to use apriori
    categorical_train = dtf_train.copy().drop(['price'],axis=1)
    categorical_test = dtf_test.copy().drop(['price'],axis=1)

    # convert years
    year_range = [0, 2000, 2008, 2020]
    year_label = ["vintage", "old", "modern"]
    year_dictionary = HelperMethods.rangeMeanDictionary(df=categorical_train,range=year_range,labels=year_label, attribute_name="yearOfRegistration")
    categorical_train['yearOfRegistration'] = HelperMethods.convertToCategorial(categorical_train,year_range,year_label,'yearOfRegistration')
    categorical_test['yearOfRegistration'] =  HelperMethods.convertToCategorial(categorical_test,year_range,year_label,'yearOfRegistration')

    # convert kilometer
    kilometer_range = [-1, 100000, 160000, 1000000]
    kilometer_label = ["low", "medium", "high"]
    kilometer_dictionary = HelperMethods.rangeMeanDictionary(df=categorical_train, range=kilometer_range, labels=kilometer_label, attribute_name="kilometer")
    categorical_train["kilometer"] = HelperMethods.convertToCategorial(categorical_train, kilometer_range, kilometer_label, "kilometer")
    categorical_test['kilometer'] = HelperMethods.convertToCategorial(categorical_test, kilometer_range, kilometer_label, 'kilometer')

    # power horse
    powerps_range = [-1, 100, 200, 300, 10000]
    powerps_label = ['low', 'medium', 'high', 'veryHigh']
    powerps_dictionary = HelperMethods.rangeMeanDictionary(categorical_train,range=powerps_range,labels=powerps_label,attribute_name="powerPS")
    categorical_train["powerPS"] = HelperMethods.convertToCategorial(categorical_train, powerps_range, powerps_label, "powerPS")
    categorical_test['powerPs'] = HelperMethods.convertToCategorial(categorical_test, powerps_range, powerps_label, 'powerPS')

    # create dictionraies to be able restore data
    convertedDictionary = {
        "yearOfRegistration": year_dictionary,
        "kilometer": kilometer_dictionary,
        "powerPS": powerps_dictionary
    }
    typeDictionary = dtf_train.dtypes.to_dict()



    # find rules
    apriori_For_null = AprioriForNull
    itemsets, rules = apriori_For_null.findRules(categorical_train, 0.3, 0.6)
    # fill values
    apriori_For_null.fill_nulls_with_apriori(self=apriori_For_null,train_data=dtf_train,categorical_data=categorical_train,rules=rules,convertedDictionary=typeDictionary,typeDictionary=typeDictionary)

    # now using "one Hot Encoding" to be able use ml model
    dtf_train = pd.get_dummies(dtf_train)
    dtf_test = pd.get_dummies(dtf_test)
    # because that test and train diffrent, taking care they have the same collumns after converting
    train_collumns = dtf_train.columns
    test_columns = dtf_test.columns
    HelperMethods.add_collumns(dtf_train,test_columns)
    HelperMethods.add_collumns(dtf_test,train_collumns)
    dtf_train = dtf_train.sort_index(axis=1)
    dtf_test = dtf_test.sort_index(axis=1)

    # start ml
    X_train = dtf_train.drop('price',axis=1)
    X_test = dtf_test.drop('price',axis=1)
    y_train = dtf_train['price']
    y_test = dtf_test['price']
    model = LinearRegression()
    prediction = model.fit(X_train,y_train).predict(X_test)
    print(r2_score(y_test,prediction))
    print("Mean Absolute Perc Error (Σ(|y - pred|/y)/n):","{:,.3f}".format(mean_absolute_percentage_error(y_test,prediction)))
    print("Mean Absolute Error (Σ|y - pred|/n):", "{:,.0f}".format(mean_absolute_error(y_test, prediction)))
    print("Root Mean Squared Error (sqrt(Σ(y - pred)^2/n)):", "{:,.0f}".format(np.sqrt(mean_squared_error(y_test, prediction))))
    return r2_score(y_test,prediction)