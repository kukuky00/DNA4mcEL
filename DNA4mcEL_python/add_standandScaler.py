from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle       #pickle模块
from sklearn.externals import joblib    #jbolib模块
from sklearn import preprocessing       #标准化数据模块

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

mydata=pd.read_table("./data/S_20181207.txt",sep=',',header=None)
y=mydata.loc[:,0]
x=mydata.loc[:,1:]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

#############################################################################################
### Logistic Regression Classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, test_size=.3, random_state=0)

##add preprocessing 1
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler1.fit_transform(x)
##add preprocessing 2
from sklearn.preprocessing import Normalizer
scaler2 = Normalizer().fit(x)
normalizedX = scaler2.transform(x)

###############################################
# # Method1 base on KNN
# knn=KNeighborsClassifier()
# knn.fit(x_train,y_train)
# #
# print ("KNN predict", knn.predict(x_test))
# # print ("Y_test_label:",y_test)
# print ("score:",knn.score(x_test,y_test))                 #score: 0.6130760986066452
#########################################
##cross-validation  =  5 to get the means
# scores = cross_val_score(knn, x, y, cv=5, scoring='accuracy')        # 0.64220
# print(scores.mean())
#############################################################################################
### LogisticRegression Classifier
# clf = LogisticRegression(penalty='l2')
# average=cross_val_score(clf, normalizedX, y, cv=cv)      #acuracy is 0.79935
# print(np.mean(average))                                 #use the MinMaxScaler acuracy is 0.8047
                                      #use the normalizedX acuracy is 0.78027

#############################################################################################
### Random Forest Classifier
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=23)              #23  27
# ###########################################################
# # clf.fit(x_train, y_train)
# # print ("Random Forest Classifier score:",clf.score(x_test, y_test))                 #0.7738
# ###########################################################
# # print(cross_val_score(clf, x, y, cv=cv))
#                                                           #acuracy is 0.7706
# average=cross_val_score(clf, normalizedX, y, cv=cv)         #MinMaxScaler acuracy is 0.7706
# print("Random Forest Classifier score:",np.mean(average))   #Normalizer 0.7603

#############################################################################################
### Decision Tree Classifier
# from sklearn import tree
# #
# clf = tree.DecisionTreeClassifier()
# # clf.fit(x_train, y_train)
# # print ("Decision Tree Classifier score:",clf.score(x_test, y_test))     #0.6998
#
#                                                           #acuracy is 0.67588
# average=cross_val_score(clf, normalizedX, y, cv=cv)         #MinMaxScaler acuracy is 0.6758
# print("Decision Tree Classifier score:",np.mean(average))   #Normalizer 0.67470

#############################################################################################
### GBDT(Gradient Boosting Decision Tree) Classifier
# from sklearn.ensemble import GradientBoostingClassifier
# #
# clf = GradientBoostingClassifier(n_estimators=200)
# # clf.fit(x_train, y_train)
# # print ("GradientBoostingClassifierscore:",clf.score(x_test, y_test))     #0.8252947481243301
#
#                                                           #acuracy is 0.8235
# average=cross_val_score(clf, normalizedX, y, cv=cv)         #MinMaxScaler acuracy is 0.82315
# print("GradientBoostingClassifier:",np.mean(average))   #Normalizer 0.8180

#############################################################################################
# # # Method1 base on SVM
# from sklearn import svm
# # clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
# # print ("score:",clf.score(x_test, y_test))            #0.8177
#
# clf = svm.SVC(kernel='linear', C=1)
#                                                           #acuracy is 0.80267
# average=cross_val_score(clf, normalizedX, y, cv=cv)         #MinMaxScaler acuracy is 0.7987
# print("svm kernel='linear' score:",np.mean(average))   #Normalizer 0.7912

from sklearn.svm import SVC
#
clf = SVC(kernel='rbf', probability=True)
# clf.fit(x_train, y_train)
# print ("SVM score:",clf.score(x_test, y_test))           #0.7952840300107181
                                                          #acuracy is 0.79517
average=cross_val_score(clf, normalizedX, y, cv=cv)         #MinMaxScaler acuracy is  0.7793
print("svm kernel='rbf' score:",np.mean(average))         #Normalizer 0.4853
