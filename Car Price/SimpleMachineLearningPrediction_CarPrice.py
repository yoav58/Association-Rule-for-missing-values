import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from NullValuesFiller import NullValuesFiller
from sklearn.preprocessing import LabelEncoder
import plotly.express as px # Data vizualization
import plotly.graph_objects as go
import Helper
import numpy as np # linear algebra
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error, mean_absolute_percentage_error,mean_absolute_error

def simple_method():
    # 1) open the files
    dtf = pd.read_csv("../autos.csv").drop('index', axis=1)
    used_car_data = dtf.copy()

    # convert the attributes that show time to an datatime object.
    used_car_data['dateCrawled'] = pd.to_datetime(used_car_data['dateCrawled'])
    used_car_data['dateCreated'] = pd.to_datetime(used_car_data['dateCreated'])
    used_car_data['lastSeen'] = pd.to_datetime(used_car_data['lastSeen'])

    # fill the null values with the must fraquant
    filling_list  = ['notRepairedDamage','vehicleType','fuelType','model','gearbox']
    NullValuesFiller.mustFrequent(used_car_data,filling_list)
    # the model
    used_car_data["model"].fillna("unknown", inplace=True)

    # delete collomuns that are not relevant or can effect the performance too much
    useless_col = ["nrOfPictures", "lastSeen", "dateCrawled", "name", "monthOfRegistration", "dateCreated", "postalCode", "seller", "offerType"]
    new_used_car = used_car_data.drop(useless_col, axis=1)

    # outliers
    clean_used_car = Helper.HelperMethods.remove_outliers(new_used_car)


    #########################################################
    # machine learning prediction
    #########################################################
    dt_c = pd.get_dummies(clean_used_car)
    dtf_train, dtf_test = train_test_split(dt_c, test_size=0.25)
    #dtf_train = pd.get_dummies(dtf_train)
    #dtf_test = pd.get_dummies(dtf_test)

    x_train = dtf_train.drop('price',axis=1)
    x_test = dtf_test.drop('price',axis=1)

    y_train = dtf_train['price']
    y_test = dtf_test['price']

    model = LinearRegression()
    prediction = model.fit(x_train,y_train).predict(x_test)
    print(r2_score(y_test,prediction))
    print("Mean Absolute Perc Error (Σ(|y - pred|/y)/n):","{:,.3f}".format(mean_absolute_percentage_error(y_test,prediction)))
    print("Mean Absolute Error (Σ|y - pred|/n):", "{:,.0f}".format(mean_absolute_error(y_test, prediction)))
    print("Root Mean Squared Error (sqrt(Σ(y - pred)^2/n)):", "{:,.0f}".format(np.sqrt(mean_squared_error(y_test, prediction))))
    return r2_score(y_test,prediction)