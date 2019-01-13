from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle       #pickle模块
from sklearn.externals import joblib    #jbolib模块
from sklearn import preprocessing       #标准化数据模块

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

def cal_rate(result):
    all_number = len(result)
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for item in range(all_number):
        disease = result[item][0]
        # print(disease)
        if disease == 1:
            if result[item][1] == 1:
                TP += 1
            else:
                FP += 1
        else:
            if result[item][1] == 0:
                TN += 1
            else:
                FN += 1
    # print TP+FP+TN+FN
    sn=float(TP) / float(TP+FN)
    sp=float(TN) / float(TN+FP)
    acc =float(TP+TN) / float(TP+FP+TN+FN)
    # mcc=float(TP*TN-FP*FN) / float((TP+FN)*(TP+FP)*(TN+FN)*(TN+FP))**0.5
    mcc = float(TP * TN - FP * FN) / math.sqrt(float((TP + FN) * (TP + FP) * (TN + FN) * (TN + FP)))

    if TP+FP == 0:
        precision = 0
    else:
        precision = float(TP) / float(TP+FP)
    TPR = float(TP) / float(TP+FN)
    TNR = float(TN) / float(FP+TN)
    FNR = float(FN) / float(TP+FN)
    FPR = float(FP) / float(FP+TN)

    print(TP,FP,TN,FN)
    print(sn,sp,acc,mcc)
    return acc,precision, TPR, TNR, FNR, FPR

mydata=pd.read_table("./data/S5_8_20181221.txt",sep=',',header=None)
######################################
y=mydata.loc[:,0]
x=mydata.loc[:,1:]

####lda : lower the dimension ####
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=10)
lda.fit(x,y)
x = lda.transform(x)

#################################################
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=0)

##add preprocessing 1
from sklearn.preprocessing import MinMaxScaler
scaler1 = MinMaxScaler(feature_range=(0, 1))
rescaledX = scaler1.fit_transform(x)
x_train1,x_test1,y_train1,y_test1=train_test_split(rescaledX,y,test_size=0.3,random_state=0)
##add preprocessing 2

from sklearn.preprocessing import Normalizer
scaler2 = Normalizer().fit(x)
normalizedX = scaler2.transform(x)
################################################
x_train2,x_test2,y_train2,y_test2=train_test_split(normalizedX,y,test_size=0.3,random_state=0)

#############################################################################################
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
cv = ShuffleSplit(n_splits=10, test_size=.3, random_state=0)

#####################################################################################
# Method1 base on SVM
from sklearn import svm
clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)

clf.fit(x_train, y_train)
print ("svm :",clf.score(x_test, y_test))
result=np.c_[clf.predict(x_test),y_test]
cal_rate(result)


##############################################
print("2#######################################################################")
##############################################
clf.fit(x_train1, y_train1)
print ("MinMaxScaler+svm:",clf.score(x_test1, y_test1))
result1=np.c_[clf.predict(x_test1),y_test1]
cal_rate(result1)
# ##############################################
print("3#######################################################################")
# ##############################################
clf.fit(x_train2, y_train2)
print ("Normalizer+svm :",clf.score(x_test2, y_test2))
result2=np.c_[clf.predict(x_test2),y_test2]
cal_rate(result2)
print(" ")
print(" ")
# #####################################################
# from sklearn.model_selection import cross_val_score
# #
# clf = svm.SVC(kernel='linear', C=1)
#
# scores = cross_val_score(clf, x, y, cv=cv)
# print("cv+score:",scores)
# print("mean+score:",scores.mean())
#
# from sklearn.svm import SVC
# #
# clf = SVC(kernel='rbf', probability=True)
# # clf.fit(x_train, y_train)
# # print ("SVM score:",clf.score(x_test, y_test))
# #
# scores = cross_val_score(clf, x, y, cv=cv)
# print("cv+score:",scores)
# print("mean+score:",scores.mean())

#############################################################################################
## LogisticRegression Classifier###
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty='l2')

clf.fit(x_train, y_train)
print ("LogisticRegression :",clf.score(x_test, y_test))
result=np.c_[clf.predict(x_test),y_test]
cal_rate(result)
##############################################
print("#######################################################################")
##############################################
clf.fit(x_train1, y_train1)
print ("MinMaxScaler+LogisticRegression:",clf.score(x_test1, y_test1))
result1=np.c_[clf.predict(x_test1),y_test1]
cal_rate(result1)
#############################################
print("#######################################################################")
##############################################
clf.fit(x_train2, y_train2)
print ("Normalizer+LogisticRegression :",clf.score(x_test2, y_test2))
result2=np.c_[clf.predict(x_test2),y_test2]
cal_rate(result2)
print(" ")
print(" ")
#######################################

# average=cross_val_score(clf, x, y, cv=cv)
# print(average)
# print(np.mean(average))

#############################################################################################
### Decision Tree Classifier
# from sklearn import tree
#
# clf = tree.DecisionTreeClassifier()
# clf.fit(x_train, y_train)
# # print ("Decision Tree Classifier score:",clf.score(x_test, y_test))     #0.6998

# average=cross_val_score(clf, x, y, cv=cv)
# print(average)
# print(np.mean(average))                                 #0.67738

#############################################################################################
######GBDT(Gradient Boosting Decision Tree) Classifier#######
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=200)
clf.fit(x_train, y_train)
print("Gradient Boosting Decision Tree score:",clf.score(x_test, y_test))
result=np.c_[clf.predict(x_test),y_test]
cal_rate(result)
##############################################
print("#######################################################################")
##############################################
clf.fit(x_train1, y_train1)
print("MinMaxScaler+Gradient Boosting Decision Tree score:",clf.score(x_test1, y_test1))
result1=np.c_[clf.predict(x_test1),y_test1]
cal_rate(result1)
##############################################
print("#######################################################################")
##############################################
clf.fit(x_train2, y_train2)
print("Normalizer+Gradient Boosting Decision Tree score:",clf.score(x_test2, y_test2))
result2=np.c_[clf.predict(x_test2),y_test2]
cal_rate(result2)
print(" ")
print(" ")
#############################################
# average=cross_val_score(clf, x, y, cv=cv)
# print('GradientBoostingClassifier:',average)
# print('GradientBoostingClassifier:',np.mean(average))                                 #0.82325
#############################################################################################

# from sklearn import naive_bayes
# model = naive_bayes.GaussianNB() # 高斯贝叶斯
# model = naive_bayes.MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)
# model = naive_bayes.BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True, class_prior=None)
#
# scores = cross_val_score(model, x, y, cv=cv)
# print("naive_bayes+score:",scores)
# print("naive_bayes+mean+score:",scores.mean())             #mean+score:0.7366
############################################################################################
from sklearn.neural_network import MLPClassifier
# # # # 定义多层感知机分类算法
clf = MLPClassifier(activation='relu', solver='adam', alpha=0.01, max_iter=200)
############################################################
clf.fit(x_train, y_train)
print ("MLPClassifier :",clf.score(x_test, y_test))
result=np.c_[clf.predict(x_test),y_test]
cal_rate(result)
##############################################
print("#######################################################################")
##############################################
clf.fit(x_train1, y_train1)
print ("MLPClassifier+Random Forest :",clf.score(x_test1, y_test1))
result1=np.c_[clf.predict(x_test1),y_test1]
cal_rate(result1)
# ##############################################
# print("#######################################################################")
# ##############################################
# clf.fit(x_train2, y_train2)
# print ("MLPClassifier+ Normalizer:",clf.score(x_test2, y_test2))
# result2=np.c_[clf.predict(x_test2),y_test2]
# cal_rate(result2)
# print(" ")
# print(" ")
###########################################
# """参数
# ---
#     hidden_layer_sizes: 元祖
#     activation：激活函数
#     solver ：优化算法{‘lbfgs’, ‘sgd’, ‘adam’}
#     alpha：L2惩罚(正则化项)参数。
# """
#
# scores = cross_val_score(model, x, y, cv=cv)
# print("naive_bayes+score:",scores)
# print("naive_bayes+mean+score:",scores.mean())             #mean+score:0.7916

#############################################################################################
### Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=23)              #23  27
# # ###########################################################
clf.fit(x_train, y_train)
print ("Random Forest :",clf.score(x_test, y_test))
result=np.c_[clf.predict(x_test),y_test]
cal_rate(result)
##############################################
print("#######################################################################")
##############################################
clf.fit(x_train1, y_train1)
print ("MinMaxScaler+Random Forest :",clf.score(x_test1, y_test1))
result1=np.c_[clf.predict(x_test1),y_test1]
cal_rate(result1)
##############################################
print("#######################################################################")
##############################################
clf.fit(x_train2, y_train2)
print ("Normalizer+Random Forest :",clf.score(x_test2, y_test2))
result2=np.c_[clf.predict(x_test2),y_test2]
cal_rate(result2)

# # ###########################################################
# average=cross_val_score(clf, x, y, cv=cv)
# print('RandomForestClassifier:',average)
# print('RandomForestClassifier:',np.mean(average))


