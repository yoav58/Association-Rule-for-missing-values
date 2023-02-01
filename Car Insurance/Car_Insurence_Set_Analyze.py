import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from Helper import HelperMethods
import plotly.graph_objects as go




####################################################################
# this file is to analyze the data set. the reason for the analyze
# is to improve the ml model and apriori performance.
###################################################################


# read the file

dtf = pd.read_csv('Car_Insurance_Claim.csv').drop('ID', axis=1)

# print the first 10 rows and see the info of the data set
print(dtf.head(10))
print(dtf.info())
print(dtf.describe())

# check the OUTCOME attribute, this attribute indicates 1 if a customer has claimed his/her loan else 0.
dtf.OUTCOME.hist()
plt.show()


# checking the amount of null value, from here we can see that credit score and annual millage have null value.
print(dtf.isna().sum())

# checking in percentage the amount of null values, from here we can see that we have around 10 percent
# of null values in CREDIT_SCORE and ANNUAL_MILEAGE thats should be enough so that fill the nulls values
# in the "Right" way will improve the machine learning model
persentage_null = dtf.isnull().sum() * 100 /len(dtf)
print(persentage_null)
print(dtf.dtypes)
# check the CREDIT_SCORE and ANNUAL_MILEAGE attributes. from what can see they look close to the normal distribution.
dtf.CREDIT_SCORE.hist()
plt.show()
dtf.ANNUAL_MILEAGE.hist()
plt.show()




# check correlation, from what we can see here, the correlations of CREDIT_SCORE and OUTCOME is quite high (-0.33),
# compared to other attributes, this can indicates that using aprioi to fill the null values in credit score
# could improve our model significantly
dtf_corr = dtf.corr(method="pearson").loc[["OUTCOME"]]
fig, ax = plt.subplots(figsize=(15,2))
sns.heatmap(dtf_corr, annot=True, fmt='.2f', cmap="YlGnBu", cbar=True, linewidths=0.5,ax=ax)
plt.show()

# checking the corrlations of all attributes, p
cat_cols = ["AGE", "GENDER",
            "RACE", "DRIVING_EXPERIENCE", "EDUCATION", "INCOME", "CREDIT_SCORE","VEHICLE_YEAR",
            "VEHICLE_TYPE"]

corr = HelperMethods.show_correlations(dtf, cat_cols)
