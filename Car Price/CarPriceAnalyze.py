import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import Helper

"""
this file is to analyze the car price data set.
"""

dtf = pd.read_csv('autoSell.csv')

# see the types of the attributes
print(dtf.dtypes)

# see statistics of each attribute
print(dtf.describe())


# see the amount of null values,as we can see
# vehicleType,gearbox,model,fuelType have a lot of null values
print(dtf.isna().sum())
pm = dtf.isnull().sum().sort_values(ascending=False) * 100 / len(dtf)
print(pm)

# see the number of unique values, since model is categorical attribute
# and has a lot of unique values, it should be better to delete him to improve performance.
unique_value_counts = dtf.nunique()
print(unique_value_counts)


# see the coorelation
Helper.HelperMethods.show_correlations(dtf,dtf.columns)

# explore the traget Attribute
dtf.price.hist(bins = 70,range=(0,16000))
print(dtf.price.count())
print(dtf["price"].isnull().sum())
plt.show()


# see the relation ship of the attributes with null values to the target feature.
fig = px.scatter(data_frame=dtf.iloc[:10000], x = "vehicleType", y = "price", title="Relationship Between vehicleType and Price")
fig.show()
fig = px.scatter(data_frame=dtf.iloc[:10000], x = "fuelType", y = "price", title="Relationship Between vehicleType and Price")
fig.show()
fig = px.scatter(data_frame=dtf.iloc[:10000], x = "gearbox", y = "price", title="Relationship Between vehicleType and Price")
fig.show()




