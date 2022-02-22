# @author: Berke Altıparmak
# You may use, distribute and modify this code under the
# terms of the Beerware license, which unfortunately won't be
# written for another century.
# ----------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42):
# <berkealtiparmak@outlook.com> wrote this file.  As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer in return.   Berke Altıparmak
# ----------------------------------------------------------------------------
#



from math import sin
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, confusion_matrix,precision_recall_curve,precision_recall_fscore_support, classification_report
from scipy.sparse import data
from sklearn.metrics._plot.precision_recall_curve import PrecisionRecallDisplay
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.utils.sparsefuncs import inplace_swap_column
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

h = 0.02  # step size in the mesh

plt.style.use('ggplot')

datacs = "CSVcsgo_round_snapshots.csv"

df = pd.read_csv(datacs, sep=',')       # DATA LOAD

fig = df.hist(figsize=(10, 10))         # DATA HISTOGRAM
[x.title.set_size(5) for x in fig.ravel()]

X = df.drop("round_winner", axis=1)   # DATA DROP round winner
X_filtered = SelectKBest(chi2, k=10).fit_transform(X, df["round_winner"]) # feature selection, top 10 features
print(X_filtered)
y = df['round_winner'] 
                 

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=40)
X_filtered_train,X_filtered_test,y_filtered_train,y_filtered_test = train_test_split(X_filtered,y,test_size=0.2,random_state=40)

classifier_RandomForest = RandomForestClassifier()       # CLASSIFIER 
clf_fit_RandomForest = classifier_RandomForest.fit(X_train, y_train)      # CLASSIFIER'DA FIT
clf_score_RandomForest = classifier_RandomForest.score(X_test, y_test)     # CLASSIFIER'DA SCORE

classifier_filtered_RandomForest = RandomForestClassifier()       # FEATURE SELECTION 
clf_filtered_fit_RandomForest = classifier_filtered_RandomForest.fit(X_filtered_train, y_filtered_train) 
clf_filtered_score_RandomForest = classifier_filtered_RandomForest.score(X_filtered_test, y_filtered_test)

classifier_KNN = KNeighborsClassifier(n_neighbors=5)       # CLASSIFIER 
clf_fit_KNN = classifier_KNN.fit(X_train, y_train)      # CLASSIFIER'DA FIT
clf_score_KNN = classifier_KNN.score(X_test, y_test)

classifier_MLP = MLPClassifier()       # CLASSIFIER 
clf_fit_MLP = classifier_MLP.fit(X_train, y_train)      # CLASSIFIER'DA FIT
clf_score_MLP = classifier_MLP.score(X_test, y_test)

classifier_NB = GaussianNB()       # CLASSIFIER 
clf_fit_NB = classifier_NB.fit(X_train, y_train)      # CLASSIFIER'DA FIT
clf_score_NB = classifier_NB.score(X_test, y_test)







classifier_LogisticRegression = LogisticRegression(max_iter=1000)    
clf_fit_LogisticRegression = classifier_LogisticRegression.fit(X_train, y_train)     
clf_score_LogisticRegression = classifier_LogisticRegression.score(X_test, y_test) 

classifier_filtered_LogisticRegression = LogisticRegression(max_iter=1000)  # FEAUTRE SELECTION
clf_filtered_fit_LogisticRegression = classifier_filtered_LogisticRegression.fit(X_filtered_train, y_filtered_train)
clf_filtered_score_LogisticRegression = classifier_filtered_LogisticRegression.score(X_filtered_test, y_filtered_test)


pred_RandomForest = list(classifier_RandomForest.predict(X_test))     # CLASSIFIER'DA PREDICT
pred_LogisticRegression = list(classifier_LogisticRegression.predict(X_test))

pred_KNN = list(classifier_KNN.predict(X_test))
pred_MLP = list(classifier_MLP.predict(X_test))
pred_NB = list(classifier_NB.predict(X_test))

pred_df = {"predicted: ": pred_RandomForest, "actual: ": y_test}
y_test_list = list(y_test)

clf_report_RandomForest = classification_report(pred_RandomForest,y_test)   # CLASSIFIER REPORT
clf_report_LogisticRegression = classification_report(pred_LogisticRegression,y_test)

clf_report_KNN = classification_report(pred_KNN,y_test)
clf_report_MLP = classification_report(pred_MLP,y_test)
clf_report_NB = classification_report(pred_NB,y_test)

conf_matrix_RandomForest = confusion_matrix(y_test, pred_RandomForest)   # CONFUSION MATRIX
conf_matrix_LogisticRegression = confusion_matrix(y_test, pred_LogisticRegression)

conf_matrix_KNN = confusion_matrix(y_test, pred_KNN)
conf_matrix_MLP = confusion_matrix(y_test, pred_MLP)
conf_matrix_NB = confusion_matrix(y_test, pred_NB)

print("                                ")
print("********************************")
print("********************************")
print('                                ')
print("random forest accuracy on test set: {}%".format(clf_score_RandomForest*100))
print('logistic regression accuracy on test set: {}%'.format(clf_score_LogisticRegression*100))
print('Feature selection random forest {}%'.format(clf_filtered_score_RandomForest*100))
print('Feature selection logistic regression {}%'.format(clf_score_LogisticRegression*100))
print('--------------------------------')
print('--------------------------------')
print('KNN: {}%'.format(clf_score_KNN*100))
print('MLP: {}%'.format(clf_score_MLP*100))
print('NB: {}%'.format(clf_score_NB*100))
print('--------------------------------')
print('--------------------------------')
print('                                ')
print("********************************")
print("********************************")
print("RANDOM FOREST Confusion matrix:", conf_matrix_RandomForest)
print("LOGISTIC REGRESSION Confusion matrix:", conf_matrix_LogisticRegression)
print("KNN Confusion matrix:", conf_matrix_KNN)
print("MLP Confusion matrix:", conf_matrix_MLP)
print("NB Confusion matrix:", conf_matrix_NB)
print('********************************')
print('********************************')
print("RANDOM FOREST Classification report:", clf_report_RandomForest)
print("LOGISTIC REGRESSION Classification report:", clf_report_LogisticRegression)
print('KNN Classification report:', clf_report_KNN)
print('MLP Classification report:', clf_report_MLP)
print('NB Classification report:', clf_report_NB)

print(pd.DataFrame(pred_df).head(20))

c_arr = [ x for x in range(100) ]

fig, axis = plt.subplots(2,2)

axis[0,0].scatter(y_test_list,pred_RandomForest, alpha=0.15)
axis[0,0].set_xlabel("actual")
axis[0,0].set_ylabel("predicted")
axis[0,0].set_title("Predicted vs Actual")

axis[0,1].plot(c_arr,pred_RandomForest[0:100], c="red")
axis[0,1].plot(c_arr,y_test_list[0:100], c="black")
axis[0,1].set_xlabel("number of samples")
axis[0,1].set_ylabel("winner pred diff")

axis[1,0].hist(pred_RandomForest, bins=10,color="blue", label="predicted")
axis[1,0].hist(y_test_list, bins=10,color="orange",alpha=0.5, label="actual")
axis[1,0].legend()

axis[1,1].bar(["RandomForest"],clf_score_RandomForest*100, color="black")
axis[1,1].bar(["LogisticRegression"],clf_score_LogisticRegression*100, color="green")
axis[1,1].set_ylabel("Accuracy %")


plt.show()







