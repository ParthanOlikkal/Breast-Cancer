# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import model_selection

#import cancer dataset 
data = pd.read_csv("https://raw.githubusercontent.com/ParthanOlikkal/Breast-Cancer/master/data.csv")
data.head()
print("Cancer dataset dimensions : {}".format(data.shape))

#Data visualization
#Histogram
data.groupby('diagnosis').size()
data.groupby('diagnosis').hist(figsize=(12, 12))

#Pairplot
sns.set(style = "ticks")
sns.pairplot(data, hue = 'diagnosis', vars = ['radius_mean', 'texture_mean', 'area_mean', 'perimeter_mean', 'smoothness_mean'] )
sns.countplot(data['diagnosis'], label = "Count")

#Heatmap(strong correlation with mean radius and mean perimeter, mean area and mean perimeter)
plt.figure(figsize=(20,10))
sns.heatmap(data.corr(), annot = True)

data.isnull().sum()
data.isna().sum()

#Train the model
X = data.iloc[:, 2:32].values #Everything other than the diagnosis is taken as X
Y = data.iloc[:, 1].values #Diagnosis is taken as Y

#Feature Scaling
from sklearn.preprocessing import LabelEncoder
Labelencoder_Y = LabelEncoder()
Y = Labelencoder_Y.fit_transform(Y)
#print(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size = 0.20, random_state = 5)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression(random_state = 0)
classifier_lr.fit(X_train, y_train)
Y_pred_lr = classifier_lr.predict(X_test)

from sklearn import metrics
print("Accuracy of Logistic Regression : ", metrics.accuracy_score(y_test, Y_pred_lr))
#97.3


#KNN algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier_knn = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski' ,p = 2)
classifier_knn.fit(X_train, y_train)
Y_pred_knn = classifier_knn.predict(X_test)

from sklearn import metrics
print("Accuracy of K Nearest Neighbors : ", metrics.accuracy_score(y_test, Y_pred_knn))
#95.6 


#SVM
from sklearn.svm import SVC
classifier_svm_linear = SVC(kernel = 'linear', random_state = 0) 
classifier_svm_linear.fit(X_train, y_train)
Y_pred_svm = classifier_svm_linear.predict(X_test)

from sklearn import metrics
print("Accuracy of SVM : ", metrics.accuracy_score(y_test, Y_pred_svm))
#96.4


from sklearn.svm import SVC
classifier_svm_rbf = SVC(kernel = 'rbf', random_state = 0)
classifier_svm_rbf.fit(X_train, y_train)
Y_pred_svm_rbf = classifier_svm_rbf.predict(X_test)

from sklearn import metrics
print("Accuracy of SVM : ", metrics.accuracy_score(y_test, Y_pred_svm_rbf))
#94.7


#Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier_naive = GaussianNB()
classifier_naive.fit(X_train, y_train)
Y_pred_naive = classifier_naive.predict(X_test)

from sklearn import metrics
print("Accuracy of Naive Bayes : ", metrics.accuracy_score(y_test, Y_pred_naive))
#94.7

#Decision Trees
from sklearn.tree import DecisionTreeClassifier
classifier_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier_tree.fit(X_train, y_train)
Y_pred_tree = classifier_tree.predict(X_test)

from sklearn import metrics
print("Accuracy of Desicion Tree : ", metrics.accuracy_score(y_test, Y_pred_tree))
#92.9


#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classifier_forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier_forest.fit(X_train, y_train)
Y_pred_forest = classifier_forest.predict(X_test)

from sklearn import metrics
print("Accuracy of Random Forest Classifier : ", metrics.accuracy_score(y_test, Y_pred_forest))
#93.8



#Spot check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(kernel = 'linear')))
models.append(('K-SVM', SVC(kernel = 'rbf')))
models.append(('Forest', RandomForestClassifier(n_estimators = 10)))

#Evaluate each model in turn
results = []
names = []
for name, model in models:
    kfold = model_selection.KFold(n_splits = 10, random_state = 0)
    cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring = 'accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name,cv_results.mean(), cv_results.std())
    print(msg)

#Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
