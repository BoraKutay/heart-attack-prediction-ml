# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 21:05:46 2021

@author: asus
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
#Feature selection
import seaborn as sns
# Scaling
from sklearn.preprocessing import StandardScaler
# Train Test Split
from sklearn.model_selection import train_test_split

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

# Metrics
from sklearn.metrics import accuracy_score, f1_score, roc_curve
from sklearn.metrics import roc_auc_score

# Cross Validation
from sklearn.model_selection import cross_val_score

heart_csv = pd.read_csv('data/heart.csv')

heart2_csv = heart_csv

'''
ANALYSIS
'''
print(heart_csv.describe())
print("CHECKING MISSING DATA")
print(heart_csv.isnull().sum())

print(heart_csv.corr()["output"].sort_values())
'''
FEATURE EXTRACTION
'''

corrmat = heart_csv.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(20,20))
#plot heat map
g=sns.heatmap(heart_csv[top_corr_features].corr(),annot=True,cmap="RdYlGn")
plt.show()


columns = np.full((corrmat.shape[0],), True, dtype=bool)
for i in range(corrmat.shape[0]):
    for j in range(i+1,corrmat.shape[0]):
        if corrmat.iloc[13,j] < -0.30:
            if columns[j]:
                columns[j] = False


y = heart2_csv['output']
try:
    X = heart_csv.drop(['output'], axis = 1)
    test_just_one = X.iloc[270]
except:
    X = heart_csv
    test_just_one = X.iloc[270]
    


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 42)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

'''
RANDOM FOREST N_ESTIMATOR SELECTION
'''
print("---------------------------------------------------------------")
print("Random Forest")
N_EST_MAX = 250
scores_rf = []
for i in range(10,N_EST_MAX,50):
    randomforest_classifier= RandomForestClassifier(n_estimators=i)
    score=cross_val_score(randomforest_classifier,X,y,cv=10)
    scores_rf.append(round(score.mean(),3))
    

plt.plot([k for k in range(10,N_EST_MAX,50)], scores_rf, linestyle="--",marker="o",color = 'blue')
j=1
n_est = []
for i in range(10,N_EST_MAX,50):
    n_est.append(i)
    plt.text(i, scores_rf[j-1], (i, scores_rf[j-1]))
    j = j + 1
    if (j > len(scores_rf)):
        break
 
    
    
plt.xticks([i for i in range(10,250,50)])
plt.xlabel('Number of Estimator (n_estimators)')
plt.ylabel(' Random Forest Scores')
plt.title('RandomForestClassifier scores for different n_estimators')
plt.show()

max_scores = max(scores_rf)
max_index = scores_rf.index(max_scores)



n = n_est[max_index]
start = time.time()
randomforest_classifier= RandomForestClassifier(n_estimators=n)
randomforest_classifier.fit(X_train,y_train)
y_pred = randomforest_classifier.predict(X_test)
end = time.time()
print("The test accuracy score of Random Forest is ", round(accuracy_score(y_test, y_pred),3))

'''
ROC CURVE OF RANDOM FOREST
'''
fpr,tpr,threshols=roc_curve(y_test,y_pred)
plt.plot([0,1],[0,1],"k--",'r+')
plt.plot(fpr,tpr,label='Random Forest')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Random Forest ROC Curve")
plt.show()

'''
f1 score of RANDOM FOREST and Confusion Matrix
'''
print('Random Forest --->',round(f1_score(y_test,y_pred),3))
print("Confusion Matrix Random Forest: \n", confusion_matrix(y_test, y_pred))
print("Execution time of Random Forest:",round(end-start,5),"seconds")

'''
KNN CLASSIFIER
'''
print("---------------------------------------------------------------")
print("KNN")
knn_scores = []
for k in range(1,15):
    knn_classifier = KNeighborsClassifier(n_neighbors = k)
    score=cross_val_score(knn_classifier,X,y,cv=10)
    knn_scores.append(round(score.mean(),3))
    
plt.plot([k for k in range(1, 15)], knn_scores,linestyle="--",marker="o", color = 'blue')
for i in range(1,15):
    plt.text(i, knn_scores[i-1], (knn_scores[i-1]))
plt.xticks([i for i in range(1, 15)])
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Scores')
plt.title('K Neighbors Classifier scores for different K values')
plt.show()

max_scores_knn = max(knn_scores)
max_index_knn = knn_scores.index(max_scores_knn)

start = time.time()
knn_classifier = KNeighborsClassifier(n_neighbors = max_index_knn)
knn_classifier.fit(X_train, y_train)
y_pred = knn_classifier.predict(X_test)
end = time.time()
print("The test accuracy score of KNN is ", round(accuracy_score(y_test, y_pred),3))

'''
ROC CURVE OF KNN
'''
fpr,tpr,threshols=roc_curve(y_test,y_pred)
plt.plot([0,1],[0,1],"k--",'r+')
plt.plot(fpr,tpr,label='KNN')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("KNN ROC Curve")
plt.show()

'''
f1 score of KNN and Confusion Matrix
'''
print('KNN --->',round(f1_score(y_test,y_pred),3))
print("Confusion Matrix KNN: \n", confusion_matrix(y_test, y_pred))
print("Execution time of KNN:",round(end-start,5),"seconds")


'''
LOGISTIC REGRESSION
'''
print("---------------------------------------------------------------")
print("Logistic Regression")
start = time.time()
logisticRegression = LogisticRegression(max_iter = 1000)
logisticRegression.fit(X_train, y_train)
y_pred = logisticRegression.predict(X_test)
end = time.time()
print("The test accuracy score of Logistic is ", round(accuracy_score(y_test, y_pred),3))

'''
ROC CURVE OF LOGISTIC REGRESSION
'''
fpr,tpr,threshols=roc_curve(y_test,y_pred)
plt.plot([0,1],[0,1],"k--",'r+')
plt.plot(fpr,tpr,label='Logistic Regression')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic ROC Curve")
plt.show()

'''
f1 score of LOGISTIC REGRESSION and Confusion Matrix
'''
print('F1 score of Logistic --->',round(f1_score(y_test,y_pred),3))
print("Confusion Matrix Logistic: \n", confusion_matrix(y_test, y_pred))
print("Execution time of Logistic Regression:",round(end-start,5),"seconds")

test_just_one_ = sc.transform(test_just_one.values.reshape(1,len(test_just_one)))

print(randomforest_classifier.predict(test_just_one_))
print(knn_classifier.predict(test_just_one_))
print(logisticRegression.predict(test_just_one_))





