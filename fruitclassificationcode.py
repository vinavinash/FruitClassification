# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 09:11:41 2021

@author: Lohith
"""
#%matplotlib inline
import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#fruits = pd.read_table('fruit_data_with_colors.txt')
fruits = pd.read_csv("fruit_data_with_colors.txt", header=0, delimiter="\t", quoting=3)
fruits.head()

print(fruits.shape)

fruits_unique = dict(zip(fruits.fruit_label.unique(),fruits.fruit_name.unique()))
print(fruits_unique)

print(fruits['fruit_name'].unique())

print(fruits.groupby('fruit_name').size())

import seaborn as sns
sns.countplot(fruits['fruit_name'],label="Count")
plt.show()

X = fruits[['mass', 'width','height','color_score']]
y = fruits['fruit_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

print(X_train)
print("\n")
print(y_train)
print("\n")
print(X_test)
print("\n")
print(y_test)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111,projection = '3d')
ax.scatter(X_train['width'], X_train['height'],X_train['color_score'], c=y_train ,marker = 'o',s=100)
ax.set_xlabel('width')
ax.set_ylabel('height')
ax.set_zlabel('color_score')
plt.show()

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, y_train) #train

KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
           metric_params=None, n_jobs=None, n_neighbors=1, p=2,
           weights='uniform')


score = knn.score(X_test,y_test)
print(score)
fruit_prediction = knn.predict([[132,5.8,8.7,0.73]])
print(fruits_unique[fruit_prediction[0]])
print(fruit_prediction[0])






