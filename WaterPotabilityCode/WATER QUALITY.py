# -*- coding: utf-8 -*-
"""
Created on Mon May 31 07:16:54 2021

@author: naveen
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier,RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from xgboost import XGBClassifier
import numpy as np
import warnings
warnings.filterwarnings('ignore')

data=pd.read_csv('water_potability.csv')
data.head()

#data.info()

""" CHECKING THE PERCENTAGE OF  MISSING VALUES IN DATASET """
for col in data.columns:
    p=(data[col].isnull().sum()/len(data))*100
    print('the column {0} have {1} percent of NAN values'.format(col,p.round(2)))
    print()
    
    # data.drop(['Sulfate'],axis=1,inplace=True)
    
""" REPLACING MISSING VALUE BY MEAN OF ALL VALUES IN RESPECTIVE COLUMN """

def replace_nan_by_mean(info):
    for col in info.columns:
        info[col].fillna(np.mean(info[col]),inplace=True)
    return info
data=replace_nan_by_mean(data)

#data.describe()

#data.info()
inp = ['ph','Hardness','Solids','Chloramines','Sulfate','Conductivity','Organic_carbon','Trihalomethanes','Turbidity']

X_train, X_test, y_train, y_test =train_test_split(data[inp],data['Potability'],test_size=0.05,random_state=42)

#"DATA VISUALIZTION"

plt.figure(figsize=(15,12))
sns.heatmap(X_train.corr(),annot=True,vmin=-1)
plt.show()

plt.figure(figsize=(18,15))
sns.pairplot(X_train)
plt.show()

plt.figure(figsize=(20,20))
for i in range(8):
    plt.subplot(4,2,(i%8)+1)
    sns.distplot(X_train[X_train.columns[i]])
    plt.title(X_train.columns[i],fontdict={'size':20,'weight':'bold'},pad=3)
plt.show()

plt.figure(figsize=(20,20))
for i in range(8):
    plt.subplot(4,2,(i%8)+1)
    sns.distplot(X_train[X_train.columns[i]])
    plt.title(X_train.columns[i],fontdict={'size':20,'weight':'bold'},pad=3)
plt.show()

#"SCALING DATA"

scaler=MinMaxScaler()
train_x_std=scaler.fit_transform(X_train)

test_x_std=scaler.transform(X_test)

models_scores=pd.DataFrame()

'''
model_log = LinearRegression()
model_log.fit(train_x_std,train_data)

log_acc=accuracy_score(test_data,model_log.predict(test_x_std))
model_acc=pd.DataFrame({'Model name':['Linear Regression'],'Accuracy':[log_acc]})
models_scores=models_scores.append(model_acc,ignore_index=True)

print('Train report of linear Regression \n',classification_report(train_data,model_log.predict(train_x_std)))
print('Test report of linear Regression \n',classification_report(test_data,model_log.predict(test_x_std)))

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(test_data,model_log.predict(test_x_std)),annot=True,)
plt.title('Confusion matrix of test data',fontdict={'size':22,'weight':'bold'})
plt.show()
'''
model_tree=DecisionTreeClassifier()
grid_tree=GridSearchCV(model_tree,param_grid={'max_depth':range(6,11)})
grid_tree.fit(X_train,y_train)
predout = grid_tree.predict(X_test)
print("Decision Tree Classified Output for Test Data")
print(predout)
tree_acc=accuracy_score(y_test,grid_tree.predict(test_x_std))
model_acc=pd.DataFrame({'Model name':['Decision Tree classifier'],'Accuracy':[tree_acc]})
models_scores=models_scores.append(model_acc,ignore_index=True)
grid_tree.best_params_

print('Train report of DecisionTreeClassifier \n',classification_report(y_train,grid_tree.predict(train_x_std)))
print('Test report of DecisionTreeClassifier \n',classification_report(y_test,grid_tree.predict(test_x_std)))

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test,grid_tree.predict(test_x_std)),annot=True)
plt.title('Confusion matrix of test data',fontdict={'size':22,'weight':'bold'})
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()

model_forest=RandomForestClassifier()
grid_forest=GridSearchCV(model_forest,param_grid={'max_depth':range(6,11)})
grid_forest.fit(X_train,y_train)
predout = grid_forest.predict(X_test)
print("Random Forest Classified Output for Test Data")
print(predout)
forest_acc=accuracy_score(y_test,grid_forest.predict(test_x_std))
model_acc=pd.DataFrame({'Model name':['Random Forest Classifier'],'Accuracy':[forest_acc]})
models_scores=models_scores.append(model_acc,ignore_index=True)
print('best param',grid_forest.best_params_)
print('best score',grid_forest.best_score_)

print('Train report of RandomForestClassifier \n',classification_report(y_train,grid_forest.predict(train_x_std)))
print('Test report of RandomForestClassifier \n',classification_report(y_test,grid_forest.predict(test_x_std)))

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test,grid_forest.predict(test_x_std)),annot=True)
plt.title('Confusion matrix of test data',fontdict={'size':22,'weight':'bold'})
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()
'''
model_xgb=XGBClassifier(n_estimators=10)
grid_xgb=GridSearchCV(model_xgb,param_grid={'n_estimators':[25,50,75,100]})
grid_xgb.fit(X_train,y_train)
predout = grid_xgb.predict(X_test)
print("XGB Classified Output for Test Data")
print(predout)
xgb_acc=accuracy_score(y_test,model_xgb.predict(test_x_std))
model_acc=pd.DataFrame({'Model name':['XGBoost'],'Accuracy':[xgb_acc]})
models_scores=models_scores.append(model_acc,ignore_index=True)

print('Train report of XGBClassifier \n',classification_report(y_train,model_xgb.predict(train_x_std)))
print('Test report of XGBClassifier \n',classification_report(y_test,model_xgb.predict(test_x_std)))

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test,model_xgb.predict(test_x_std)),annot=True)
plt.title('Confusion matrix of test data',fontdict={'size':22,'weight':'bold'})
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()
'''
model_neighbor=KNeighborsClassifier()
grid_neighbor=GridSearchCV(model_neighbor,param_grid={'n_neighbors':range(4,12)})
grid_neighbor.fit(X_train,y_train)
predout = grid_neighbor.predict(X_test)
print("KNN Classified Output for Test Data")
print(predout)
neighbors_acc=accuracy_score(y_test,grid_neighbor.predict(test_x_std))
model_acc=pd.DataFrame({'Model name':['KNeighborsClassifier'],'Accuracy':[neighbors_acc]})
models_scores=models_scores.append(model_acc,ignore_index=True)
grid_neighbor.best_params_

print('Train report of KneighborsClassifier \n',classification_report(y_train,grid_neighbor.predict(train_x_std)))
print('Test report of KneighborsClassifier \n',classification_report(y_test,grid_neighbor.predict(test_x_std)))

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test,grid_neighbor.predict(test_x_std)),annot=True)
plt.title('Confusion matrix of test data',fontdict={'size':22,'weight':'bold'})
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()

model_svc=SVC(C=2)
model_svc.fit(X_train,y_train)
predout = model_svc.predict(X_test)
print("SVC Classified Output for Test Data")
print(predout)
svc_acc=accuracy_score(y_test,grid_neighbor.predict(test_x_std))
model_acc=pd.DataFrame({'Model name':['SVC'],'Accuracy':[svc_acc]})
models_scores=models_scores.append(model_acc,ignore_index=True)

print('Train report of SVClassifier \n',classification_report(y_train,model_svc.predict(train_x_std)))
print('Test report of SVClassifier \n',classification_report(y_test,model_svc.predict(test_x_std)))

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test,model_svc.predict(test_x_std)),annot=True)
plt.title('Confusion matrix of test data',fontdict={'size':22,'weight':'bold'})
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()

model_adaboost=AdaBoostClassifier(n_estimators=70)
model_adaboost.fit(X_train,y_train)
predout = model_adaboost.predict(X_test)
print("Adaboost Classified Output for Test Data")
print(predout)
adaboost_acc=accuracy_score(y_test,model_adaboost.predict(test_x_std))
model_acc=pd.DataFrame({'Model name':['Adaboost'],'Accuracy':[adaboost_acc]})
models_scores=models_scores.append(model_acc,ignore_index=True)

print('Train report of AdaboostClassifier \n',classification_report(y_train,model_adaboost.predict(train_x_std)))
print('Test report of ADAboostClassifier \n',classification_report(y_test,model_adaboost.predict(test_x_std)))

plt.figure(figsize=(10,8))
sns.heatmap(confusion_matrix(y_test,model_adaboost.predict(test_x_std)),annot=True)
plt.title('Confusion matrix of test data',fontdict={'size':22,'weight':'bold'})
plt.xlabel('Predicted value')
plt.ylabel('Actual value')
plt.show()

models_scores.sort_values(by=['Accuracy'],ascending=False)
