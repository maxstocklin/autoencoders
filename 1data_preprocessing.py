import xgboost as xgb
from xgboost import XGBClassifier

import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, KFold


warnings.filterwarnings("ignore")

# Extract feature and target arrays
dataset = pd.read_csv('creditcard.csv')

'''
print(dataset.head())
print(dataset.info())
print(dataset.describe())
'''
# Check missing values
print(dataset.isnull().sum())

# Plot Survivors data
dataset.Class.value_counts().plot(kind='bar')
plt.show()
exit()

# Droping Cabin du to too many missing values
dataset = dataset.drop('Cabin', axis=1)


# Check categorical data
print(dataset.dtypes)

'''
sns.distplot(dataset['PassengerId'],kde = False)
plt.show()

# Plot Survivors data
sns.distplot(dataset['Survived'],kde = False)
plt.show()
dataset.Survived.value_counts().plot(kind='bar')
# plt.show()

sns.distplot(dataset['Pclass'],kde = False)
plt.show()

sns.distplot(dataset['Age'],kde = False)
plt.show()

sns.distplot(dataset['SibSp'],kde = False)
plt.show()

sns.distplot(dataset['Parch'],kde = False)
plt.show()

sns.distplot(dataset['Fare'],kde = False)
plt.show()
'''

fig, ax = plt.subplots(figsize=[10, 6])
sns.boxplot(
    data=dataset,
    y='Age',
    x='Survived'
)
ax.set_title('Boxplot, Survivors by Age')
plt.show()

Survived = dataset.loc[dataset.Survived == 1]
NotSurvived = dataset.loc[dataset.Survived == 0]
ax = sns.kdeplot(Survived.Age,
                  shade=True,shade_lowest=False, label = "Survived")
ax = sns.kdeplot(NotSurvived.Age,
                  shade=True,shade_lowest=False, label = "Died")
plt.show()

ax = sns.violinplot(x="Survived", y="Age", data=dataset, hue="Survived")
plt.show()
