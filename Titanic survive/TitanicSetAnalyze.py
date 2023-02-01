import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stat
import warnings

import Helper

"""
this file is to analyze the titanic data set
"""

dtf = pd.read_csv('Titanic-Dataset.csv')

# see the data types
print(dtf.describe())


# see the corrleations of the data, as we can see
# the sex and pclass has the must correlation with the chance to
# survive
#Helper.HelperMethods.show_correlations(dtf,dtf.columns)

# see the number of null values, from what it looks age, and cabin
# have significant amount of null values
print(dtf.isna().sum())
persentage_null = dtf.isnull().sum() * 100 /len(dtf)
print(persentage_null)

# look at Cabin attribute, there a lot of possible values
# and the amount of value count for each value is very low, also we
# cannot to convert this to bins, so it should be better to delete this attribute.
print(dtf['Cabin'].value_counts())

# look at the Age attribute.
print(dtf['Age'].value_counts())
dtf['Age'].plot.hist(bins=3,alpha=0.5)
plt.show()

# check outliers at age
plt.subplot(2,1,1)
sns.boxplot(dtf['Age'])
plt.subplot(2,1,2)
sns.boxplot(dtf['Fare'])
plt.show()


# explore the Survived (target) attribute,
# from
print(dtf['Survived'].value_counts())
sns.countplot(x=dtf['Survived'])
plt.show()
