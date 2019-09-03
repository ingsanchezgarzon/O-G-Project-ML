#!/usr/bin/env python
# coding: utf-8

import numpy as np # linear algebra
import pandas as pd # 
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()

#Importing the auxiliar and preprocessing librarys 
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import accuracy_score

normalized = pd.read_csv('normalized.csv', delimiter=',')
#This file contains the information of the DB without the comments of the causes 
normalized.head()


# Select numeric columns only
numeric_cols = [cname for cname in normalized.columns if normalized[cname].dtype in ['int64', 'float64']]
X = normalized[numeric_cols].copy()

y = normalized.Amount
X.drop(['Amount'], axis=1, inplace=True)


from sklearn.model_selection import train_test_split
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)


from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

my_pipeline = Pipeline(steps=[
    ('preprocessor', SimpleImputer()),
    ('model', RandomForestRegressor(n_estimators=50, random_state=0))
])


# In[13]:


#Write a scoring function
from sklearn.model_selection import cross_val_score

# Multiply by -1 since sklearn calculates *negative* MAE
scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=5,
                              scoring='neg_mean_absolute_error')

print("Average MAE score:", scores.mean())


def get_score(n_estimators):
    """Return the average MAE over 3 CV folds of random forest model.
        Keyword argument:
    n_estimators -- the number of trees in the forest
    """
    my_pipeline = Pipeline(steps=[
        ('preprocessor', SimpleImputer()),
        ('model', RandomForestRegressor(n_estimators, random_state=0))
    ])
    scores = -1 * cross_val_score(my_pipeline, X, y,
                              cv=3,
                              scoring='neg_mean_absolute_error')
    # Replace this body with your own code
    return scores.mean()


results = {}
for i in range (50,750,50):
    results[i] = get_score(i) 


plt.plot(results.keys(), results.values())
plt.show()

n_estimators_best = min(results, key=results.get)
print(n_estimators_best)

# best model: RandomForestRegressor(n_estimators=50, random_state=0)
