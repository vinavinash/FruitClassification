# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 18:02:09 2021

@author: Lohith
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(color_codes=True)



data_frame = pd.read_csv("pimadata.csv")



print(data_frame.shape)

print(data_frame.head(5))

print(data_frame.tail(5))


print(data_frame.isnull().values.any())


def plot_corr(data_frame, size=11):
    """
    Function plots a graphical correlation matrix for each pair of columns in the dataframe.

    Input:
        data_frame: pandas DataFrame
        size: vertical and horizontal size of the plot

    Displays:
        matrix of correlation between columns.  Blue-cyan-yellow-red-darkred => less to more correlated
                                                0 ------------------>  1
                                                Expect a darkred line running from top left to bottom right
    """

    corr = data_frame.corr()    # data frame correlation function
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)   # color code the rectangles by correlation value
    plt.xticks(range(len(corr.columns)), corr.columns)  # draw x tick marks
    plt.yticks(range(len(corr.columns)), corr.columns)  # draw y tick marks


#plot_corr(data_frame)

print('The skin and thickness columns are correlated 1 to 1. Dropping the skin column')

del data_frame['skin']

print(data_frame.corr())

#plot_corr(data_frame)

print('The correlations look good. There appear to be no coorelated columns.')



print(data_frame.head(5))

diabetes_map = {True : 1, False : 0}

data_frame['diabetes'] = data_frame['diabetes'].map(diabetes_map)


data_frame.isnull().values.any()


# Check class distribution


num_obs = len(data_frame)
num_true = len(data_frame.loc[data_frame['diabetes'] == 1])
num_false = len(data_frame.loc[data_frame['diabetes'] == 0])
print("Number of True cases:  {0} ({1:2.2f}%)".format(num_true, ((1.00 * num_true)/(1.0 * num_obs)) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(num_false, ((1.00 * num_false)/(1.0 * num_obs)) * 100))

# We check to ensure we have the the desired 70% train, 30% test split of the data

from sklearn.model_selection import train_test_split

feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = data_frame[feature_col_names].values     # predictor feature columns (8 X m)
y = data_frame[predicted_class_names].values # predicted class (1=true, 0=false) column (1 X m)

split_test_size = 0.30

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_test_size, random_state=42) 


trainval = (1.0 * len(X_train)) / (1.0 * len(data_frame.index))
testval = (1.0 * len(X_test)) / (1.0 * len(data_frame.index))
print("{0:0.2f}% in training set".format(trainval * 100))
print("{0:0.2f}% in test set".format(testval * 100))


# veryfying predicted value was split correctly


print("Original True  : {0} ({1:0.2f}%)".format(len(data_frame.loc[data_frame['diabetes'] == 1]), (len(data_frame.loc[data_frame['diabetes'] == 1])/len(data_frame.index)) * 100.0))
print("Original False : {0} ({1:0.2f}%)".format(len(data_frame.loc[data_frame['diabetes'] == 0]), (len(data_frame.loc[data_frame['diabetes'] == 0])/len(data_frame.index)) * 100.0))
print("")
print("Training True  : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 1]), (len(y_train[y_train[:] == 1])/len(y_train) * 100.0)))
print("Training False : {0} ({1:0.2f}%)".format(len(y_train[y_train[:] == 0]), (len(y_train[y_train[:] == 0])/len(y_train) * 100.0)))
print("")
print("Test True      : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 1]), (len(y_test[y_test[:] == 1])/len(y_test) * 100.0)))
print("Test False     : {0} ({1:0.2f}%)".format(len(y_test[y_test[:] == 0]), (len(y_test[y_test[:] == 0])/len(y_test) * 100.0)))



# Hidden Missing Values


print("# rows in dataframe {0}".format(len(data_frame)))
print("# rows missing glucose_conc: {0}".format(len(data_frame.loc[data_frame['glucose_conc'] == 0])))
print("# rows missing diastolic_bp: {0}".format(len(data_frame.loc[data_frame['diastolic_bp'] == 0])))
print("# rows missing thickness: {0}".format(len(data_frame.loc[data_frame['thickness'] == 0])))
print("# rows missing insulin: {0}".format(len(data_frame.loc[data_frame['insulin'] == 0])))
print("# rows missing bmi: {0}".format(len(data_frame.loc[data_frame['bmi'] == 0])))
print("# rows missing diab_pred: {0}".format(len(data_frame.loc[data_frame['diab_pred'] == 0])))
print("# rows missing age: {0}".format(len(data_frame.loc[data_frame['age'] == 0])))



# Impute with the mean

from sklearn.impute import SimpleImputer

#Impute with mean all 0 readings
fill_0 = SimpleImputer(missing_values=0, strategy="mean")

X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)


# Naive bayes Algorithm

from sklearn.naive_bayes import GaussianNB

# create Gaussian Naive Bayes model object and train it with the data
nb_model = GaussianNB()

nb_model.fit(X_train, y_train.ravel())



# this returns array of predicted results
prediction_from_trained_data = nb_model.predict(X_train)




from sklearn import metrics

nb_accuracy_train = metrics.accuracy_score(y_train, prediction_from_trained_data)

nb_precision_train = metrics.precision_score(y_train, prediction_from_trained_data)

nb_recall_train = metrics.recall_score(y_train, prediction_from_trained_data)

nb_flscore_train = metrics.f1_score(y_train, prediction_from_trained_data)

print ("Accuracy of our Naive Bayes model for Trained Data is : {0:.4f}".format(nb_accuracy_train))

print ("Precision of our Naive Bayes model for Trained Data is : {0:.4f}".format(nb_precision_train))

print ("Recall of our Naive Bayes model for Trained Data is : {0:.4f}".format(nb_recall_train))

print ("F1-score of our Naive Bayes model for Trained Data is : {0:.4f}".format(nb_flscore_train))



# this returns array of predicted results from test_data
prediction_from_test_data = nb_model.predict(X_test)

nb_accuracy_test = metrics.accuracy_score(y_test, prediction_from_test_data)

nb_precision_test = metrics.precision_score(y_test, prediction_from_test_data)

nb_recall_test = metrics.recall_score(y_test, prediction_from_test_data)

nb_f1score_test = metrics.f1_score(y_test, prediction_from_test_data)

print ("Accuracy of our Naive Bayes model for Test Data is: {0:0.4f}".format(nb_accuracy_test))

print ("Precision of our Naive Bayes model for Test Data is: {0:0.4f}".format(nb_precision_test))

print ("Recall of our Naive Bayes model for Test Data is: {0:0.4f}".format(nb_recall_test))

print ("F1-score of our Naive Bayes model for Test Data is: {0:0.4f}".format(nb_f1score_test))



print ("Confusion Matrix for Naive Bayes is")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.confusion_matrix(y_test, prediction_from_test_data, labels=[1, 0])))


print ("Classification Report for Naive Bayes is")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.classification_report(y_test, prediction_from_test_data, labels=[1, 0])))



# Random Forest Algorithm


from sklearn.ensemble import RandomForestClassifier

# Create a RandomForestClassifier object

rf_model = RandomForestClassifier(random_state=42)

rf_model.fit(X_train, y_train.ravel())


rf_predict_train = rf_model.predict(X_train)

#get accuracy
rf_accuracy_train = metrics.accuracy_score(y_train, rf_predict_train)

rf_precision_train = metrics.precision_score(y_train, rf_predict_train)

rf_recall_train = metrics.recall_score(y_train, rf_predict_train)

rf_flscore_train = metrics.f1_score(y_train, rf_predict_train)

#print accuracy
print ("Accuracy of our Random Forest model for Train Data: {0:.4f}".format(rf_accuracy_train))

print ("Precision of our Random Forest model for Train Data: {0:.4f}".format(rf_precision_train))

print ("Recall of our Random Forest model for Train Data: {0:.4f}".format(rf_recall_train))

print ("F1Score of our Random Forest model for Train Data: {0:.4f}".format(rf_flscore_train))





rf_predict_test = rf_model.predict(X_test)

#get accuracy
rf_accuracy_testdata = metrics.accuracy_score(y_test, rf_predict_test)

rf_precision_testdata = metrics.precision_score(y_test, rf_predict_test)

rf_recall_testdata = metrics.recall_score(y_test, rf_predict_test)

rf_f1score_testdata = metrics.f1_score(y_test, rf_predict_test)

#print accuracy
print ("Accuracy of our Random Forest model for Test Data: {0:.4f}".format(rf_accuracy_testdata))

print ("Precision of our Random Forest model for Test Data: {0:.4f}".format(rf_precision_testdata))

print ("Recall of our Random Forest model for Test Data: {0:.4f}".format(rf_recall_testdata))

print ("F1-score of our Random Forest model for Test Data: {0:.4f}".format(rf_f1score_testdata))




print ("Confusion Matrix for Random Forest is")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.confusion_matrix(y_test, rf_predict_test, labels=[1, 0])))

print ("")

print ("Classification Report for Random Forest is\n")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.classification_report(y_test, rf_predict_test, labels=[1, 0])))


# Logistic Regression


from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression(solver='liblinear', random_state=0)
lr_model.fit(X_train, y_train.ravel())

lr_predict_train = lr_model.predict(X_train)

#get accuracy
lr_accuracy_train = metrics.accuracy_score(y_train, lr_predict_train)

lr_precision_train = metrics.precision_score(y_train, lr_predict_train)

lr_recall_train = metrics.recall_score(y_train, lr_predict_train)

lr_flscore_train = metrics.f1_score(y_train, lr_predict_train)

#print accuracy
print ("Accuracy of our Logistic Regression model for Train Data: {0:.4f}".format(lr_accuracy_train))

print ("Precision of our Logistic Regression model for Train Data: {0:.4f}".format(lr_precision_train))

print ("Recall of our Logistic Regression model for Train Data: {0:.4f}".format(lr_recall_train))

print ("F1Score of our Logistic Regression model for Train Data: {0:.4f}".format(lr_flscore_train))




lr_predict_test = lr_model.predict(X_test)

#get accuracy
lr_accuracy_testdata = metrics.accuracy_score(y_test, lr_predict_test)

lr_precision_testdata = metrics.precision_score(y_test, lr_predict_test)

lr_recall_testdata = metrics.recall_score(y_test, lr_predict_test)

lr_f1score_testdata = metrics.f1_score(y_test, lr_predict_test)

#print accuracy
print ("Accuracy of our Logistic Regression model for Test Data: {0:.4f}".format(lr_accuracy_testdata))

print ("Precision of our Logistic Regression model for Test Data: {0:.4f}".format(lr_precision_testdata))

print ("Recall of our Logistic Regression model for Test Data: {0:.4f}".format(lr_recall_testdata))

print ("F1-score of our Logistic Regression model for Test Data: {0:.4f}".format(lr_f1score_testdata))




# training metrics
#print ("Accuracy of our Logistic Regression model for Test Data : {0:.4f}".format(metrics.accuracy_score(y_test, lr_predict_test)))

print ("Confusion Matrix for Logistic Regression is")

print (metrics.confusion_matrix(y_test, lr_predict_test, labels=[1, 0]))

print ("")

print ("Classification Report for Logistic Regression is")

print (metrics.classification_report(y_test, lr_predict_test, labels=[1, 0]))


#  SVM

from sklearn.svm import SVC
# Create a svm object
svm_model = SVC(kernel='linear', C=1, random_state=42)

svm_model.fit(X_train, y_train.ravel())

# this returns array of predicted results
prediction_from_trained_data = svm_model.predict(X_train)

from sklearn import metrics

svm_accuracy_train = metrics.accuracy_score(y_train, prediction_from_trained_data)

svm_precision_train = metrics.precision_score(y_train, prediction_from_trained_data)

svm_recall_train = metrics.recall_score(y_train, prediction_from_trained_data)

svm_f1score_train = metrics.f1_score(y_train, prediction_from_trained_data)

print ("Accuracy of our SVM model for Train Data is : {0:.4f}".format(svm_accuracy_train))

print ("Precision of our SVM model for Train Data is : {0:.4f}".format(svm_precision_train))

print ("Recall of our SVM model for Train Data is : {0:.4f}".format(svm_recall_train))

print ("F1-score of our SVM model for Train Data is : {0:.4f}".format(svm_f1score_train))


svm_predict_test = svm_model.predict(X_test)

#get accuracy
svm_accuracy_testdata = metrics.accuracy_score(y_test, svm_predict_test)

svm_precision_testdata = metrics.precision_score(y_train, prediction_from_trained_data)

svm_recall_testdata = metrics.recall_score(y_train, prediction_from_trained_data)

svm_flscore_testdata = metrics.f1_score(y_train, prediction_from_trained_data)

#print accuracy
print ("Accuracy of our SVM model for Test Data is: {0:.4f}".format(svm_accuracy_testdata))

print ("Precision of our SVM model for Test Data is: {0:.4f}".format(svm_precision_testdata))

print ("Recall of our SVM model for Test Data is: {0:.4f}".format(svm_recall_testdata))

print ("F1-score of our SVM model for Test Data is: {0:.4f}".format(svm_flscore_testdata))



print ("Confusion Matrix for Support Vector Machine is")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.confusion_matrix(y_test, svm_predict_test, labels=[1, 0])))

print ("")

print ("Classification Report for Support Vector Machine is\n")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.classification_report(y_test, svm_predict_test, labels=[1, 0])))


#Neural Network
from sklearn.neural_network import MLPClassifier
# Create a MLP object
ann_model = MLPClassifier(hidden_layer_sizes=(13,13,13),max_iter=500, random_state=42)

ann_model.fit(X_train, y_train.ravel())

# this returns array of predicted results
prediction_from_trained_data = ann_model.predict(X_train)

from sklearn import metrics

nn_accuracy_train = metrics.accuracy_score(y_train, prediction_from_trained_data)

nn_precision_train = metrics.precision_score(y_train, prediction_from_trained_data)

nn_recall_train = metrics.recall_score(y_train, prediction_from_trained_data)

nn_f1score_train = metrics.f1_score(y_train, prediction_from_trained_data)

print ("Accuracy of our ANN model for Train Data is : {0:.4f}".format(nn_accuracy_train))

print ("Precision of our ANN model for Train Data is : {0:.4f}".format(nn_precision_train))

print ("Recall of our ANN model for Train Data is : {0:.4f}".format(nn_recall_train))

print ("F1-score of our ANN model for Train Data is : {0:.4f}".format(nn_f1score_train))

ann_predict_test = ann_model.predict(X_test)

#get accuracy
nn_accuracy_testdata = metrics.accuracy_score(y_test, ann_predict_test)

nn_precision_testdata = metrics.precision_score(y_train, prediction_from_trained_data)

nn_recall_testdata = metrics.recall_score(y_train, prediction_from_trained_data)

nn_flscore_testdata = metrics.f1_score(y_train, prediction_from_trained_data)

#print accuracy
print ("Accuracy of our ANN model for Test Data is: {0:.4f}".format(nn_accuracy_testdata))

print ("Precision of our ANN model for Test Data is: {0:.4f}".format(nn_precision_testdata))

print ("Recall of our ANN model for Test Data is: {0:.4f}".format(nn_recall_testdata))

print ("F1-Score of our ANN model for Test Data is: {0:.4f}".format(nn_flscore_testdata))

print ("Confusion Matrix for Artificial Neural Network is")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.confusion_matrix(y_test, ann_predict_test, labels=[1, 0])))

print ("")

print ("Classification Report for Artificial Neural Network is\n")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.classification_report(y_test, ann_predict_test, labels=[ 1,0])))


# Decision Tree

from sklearn.tree import DecisionTreeClassifier

# create Gaussian Naive Bayes model object and train it with the data
dt_model = DecisionTreeClassifier(random_state=42)

dt_model.fit(X_train, y_train.ravel())


# this returns array of predicted results
prediction_from_trained_data = dt_model.predict(X_train)

from sklearn import metrics

dt_accuracy_train = metrics.accuracy_score(y_train, prediction_from_trained_data)

dt_precision_train = metrics.precision_score(y_train, prediction_from_trained_data)

dt_recall_train = metrics.recall_score(y_train, prediction_from_trained_data)

dt_flscore_train = metrics.f1_score(y_train, prediction_from_trained_data)

print ("Accuracy of our DT model for Train Model is : {0:.4f}".format(dt_accuracy_train))

print ("Precision of our DT model for Train Model is : {0:.4f}".format(dt_precision_train))

print ("Recall of our DT model for Train Model is : {0:.4f}".format(dt_recall_train))

print ("F1-Score of our DT model for Train Model is : {0:.4f}".format(dt_flscore_train))

dt_predict_test = dt_model.predict(X_test)

#get accuracy
dt_accuracy_testdata = metrics.accuracy_score(y_test, dt_predict_test)

dt_precision_testdata = metrics.precision_score(y_test, dt_predict_test)

dt_recall_testdata = metrics.recall_score(y_test, dt_predict_test)

dt_f1score_testdata = metrics.f1_score(y_test, dt_predict_test)


#print accuracy
print ("Accuracy of our DT model for Test Model is: {0:.4f}".format(dt_accuracy_testdata))

print ("Precision of our DT model for Test Model is: {0:.4f}".format(dt_precision_testdata))

print ("Recall of our DT model for Test Model is: {0:.4f}".format(dt_recall_testdata))

print ("F1-Score of our DT model for Test Model is: {0:.4f}".format(dt_f1score_testdata))




print ("Confusion Matrix for Decision Tree is")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.confusion_matrix(y_test, dt_predict_test, labels=[1, 0])))

print ("")

print ("Classification Report for Decison Tree Model is\n")

# labels for set 1=True to upper left and 0 = False to lower right
print ("{0}".format(metrics.classification_report(y_test, dt_predict_test, labels=[ 1,0])))

from sklearn.metrics import roc_curve, auc

Y_nb_score = nb_model.predict_proba(X_test)

Y_lr_score = lr_model.decision_function(X_test)

Y_rf_score = rf_model.predict_proba(X_test)

Y_svm_score = svm_model.decision_function(X_test)

Y_nn_score = ann_model.predict_proba(X_test)

Y_dt_score =dt_model.predict_proba(X_test)

fpr_nb, tpr_nb, thresholds_nb = roc_curve(y_test, Y_nb_score[:, 1])

fpr_lr, tpr_lr, thresholds_lr = roc_curve(y_test, Y_lr_score)

fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, Y_rf_score[:, 1])

fpr_svm, tpr_svm, thresholds_svm = roc_curve(y_test, Y_svm_score)

fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, Y_nn_score[:, 1])

fpr_dt, tpr_dt, thresholds_dt = roc_curve(y_test, Y_dt_score[:, 1])


import matplotlib.pyplot as plt
plt.style.use('seaborn')

# plot roc curves
plt.plot(fpr_nb, tpr_nb, linestyle='--',color='orange', label='Naive Bayes')
plt.plot(fpr_lr, tpr_lr, linestyle='--',color='green', label='Logistic Regression')
plt.plot(fpr_rf, tpr_rf, linestyle='--',color='blue', label='Random Forest')
plt.plot(fpr_svm, tpr_svm, linestyle='--',color='black', label='Support Vector Machine')
plt.plot(fpr_nn, tpr_nn, linestyle='--',color='red', label='Neural Network')
plt.plot(fpr_dt, tpr_dt, linestyle='--',color='yellow', label='Decision Tree')

# title
plt.title('AUC_ROC curve')
# x label
plt.xlabel('False Positive Rate')
# y label
plt.ylabel('True Positive rate')

plt.legend(loc='best')
plt.savefig('AUC_ROC',dpi=300)
plt.show();
