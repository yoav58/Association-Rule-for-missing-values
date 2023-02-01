import pandas as pd
from matplotlib import pyplot as plt




####################################################################
# this file is to analyze the data set. the reason for the analyze
# is to improve the ml model and apriori performance.
####################################################################


# read the file
import Helper

dtf_train = pd.read_csv('train.csv').drop(['ID','Segmentation'], axis=1)
dtf_test = pd.read_csv('test.csv').drop('ID',axis=1)


# start analyze the train
print(dtf_train.describe())

# see the types of the attributes
print(dtf_train.dtypes)


# see the amount of null values, from what it look Work_Experience have 10 percent of missing values,
# which could be enough to impact our model
print(dtf_train.isna().sum())
persentage_null = dtf_train.isnull().sum() * 100 /len(dtf_train)
print(persentage_null)

# look at the distribution of the target attribute.
#dtf_train.Spending_Score.hist()
# plt.show()

# look at Work_Experience and Family_Size hist to see how convert them to categorical.
dtf_train.Work_Experience.hist()
plt.show()
#dtf_train.Family_Size.hist()
#plt.show()
#dtf_train.Age.hist
#plt.show()


# since some of the numerical attributes should be converted, look if we there low range of values to be able
# to use astype() method
print(dtf_train.nunique())

# correlations, as it look, 'Work_Experience' have low corr with the other attributes, it could indicate that the
# apriori algorithm would not improve the ml model significantly,but still Even if the correlation between variables is
# low, association rules may still be able to identify meaningful patterns
corr = ['Gender','Ever_Married','Age','Graduated','Profession','Work_Experience','Spending_Score','Family_Size','Var_1']
#Helper.HelperMethods.show_correlations(data_frame=dtf_train,categorical_columns=corr)