
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
features = pd.read_csv("Reading_month.csv")
#features=pd.get_dummies(features)
#features.head(10)
labels=np.array(features['Units'])
#print(labels)
features=features.drop('Units',axis=1)
features_list=list(features.columns)
print(features_list)
feat=np.array(features)
print('Total Features:',feat.shape)
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 42)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(train_features, train_labels);
predictions = rf.predict(test_features)
print(test_labels)
print(predictions)
errors = abs(predictions - test_labels)
print(errors)
mape = 100 * (errors / test_labels)
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

