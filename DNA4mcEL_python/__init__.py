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
##############################################
# clf.fit(x_train, y_train)
# print ("score:",clf.score(x_test, y_test))                #score: 0.8102
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
# print(cross_val_score(clf, x, y, cv=cv))

#############################################################################################
### Random Forest Classifier
# from sklearn.ensemble import RandomForestClassifier
# clf = RandomForestClassifier(n_estimators=23)              #23  27
# ###########################################################
# clf.fit(x_train, y_train)
# # print ("Random Forest Classifier score:",clf.score(x_test, y_test))                 #0.7738
# ###########################################################
# average=cross_val_score(clf, x, y, cv=cv)
# print(average)
# print(np.mean(average))                                 #0.7702

#############################################################################################
### Decision Tree Classifier
# from sklearn import tree
#
# clf = tree.DecisionTreeClassifier()
# clf.fit(x_train, y_train)
# # print ("Decision Tree Classifier score:",clf.score(x_test, y_test))     #0.6998
#
# average=cross_val_score(clf, x, y, cv=cv)
# print(average)
# print(np.mean(average))                                 #0.67738

#############################################################################################
### GBDT(Gradient Boosting Decision Tree) Classifier
# from sklearn.ensemble import GradientBoostingClassifier
#
# clf = GradientBoostingClassifier(n_estimators=200)
# # clf.fit(x_train, y_train)
# # print ("Gradient Boosting Decision Tree score:",clf.score(x_test, y_test))     #0.8252947481243301
#
# average=cross_val_score(clf, x, y, cv=cv)
# print(average)
# print(np.mean(average))                                 #0.82325
#############################################################################################
#############################################################################################
# # # Method1 base on SVM
# from sklearn import svm
# # clf = svm.SVC(kernel='linear', C=1).fit(x_train, y_train)
# # print ("score:",clf.score(x_test, y_test))            #0.8177
# #
# from sklearn.model_selection import cross_val_score
# #
# clf = svm.SVC(kernel='linear', C=1)
# #
# scores = cross_val_score(clf, x, y, cv=cv)
# print("cv+score:",scores)
# print("mean+score:",scores.mean())                      #mean+score: 0.8026

from sklearn.svm import SVC
#
clf = SVC(kernel='rbf', probability=True)
# clf.fit(x_train, y_train)
# print ("SVM score:",clf.score(x_test, y_test))           #0.7952840300107181
#
scores = cross_val_score(clf, x, y, cv=cv)
print("cv+score:",scores)
print("mean+score:",scores.mean())                      #mean+score: 0.7951



# from sklearn.cross_validation import cross_val_score # K折交叉验证模块
# 第四步：建立测试参数集,调参，优化模型
# k_range = range(1, 31)
# k_scores = []
#
# #藉由迭代的方式来计算不同参数对模型的影响，并返回交叉验证后的平均准确率
# for k in k_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#     scores = cross_val_score(knn, x, y, cv=10, scoring='accuracy')     # 分类
#     # loss = -cross_val_score(knn, X, y, cv=10, scoring='mean_squared_error')    # 回归
#     k_scores.append(scores.mean())

# #可视化数据
# plt.plot(k_range, k_scores)
# plt.xlabel('Value of K for KNN')
# plt.ylabel('Cross-Validated Accuracy')
# plt.show()

# # 第五步： 保存模型
# # 保存Model 方法一：(注:save文件夹要预先建立，否则会报错)  pickle 方法
# with open('Model/Pickle/KNeighborsClassifier.pickle', 'wb') as f:
#     pickle.dump(knn, f)
#
# #读取Model
# with open('Model/Pickle/KNeighborsClassifier.pickle', 'rb') as f:
#     clf2 = pickle.load(f)
#     print(clf2.predict(iris_X))         #测试读取后的Model
#
#
# #保存Model 方法二：(注:save文件夹要预先建立，否则会报错) joblib 方法  优点：效率高
# joblib.dump(knn, 'Model/Joblib/KNeighborsClassifier.pkl')
#
# #读取Model
# clf3 = joblib.load('Model/Joblib/KNeighborsClassifier.pkl')
# print(clf3.predict(iris_X))           #测试读取后的Model

# ('score:', '0.6265292981326465')