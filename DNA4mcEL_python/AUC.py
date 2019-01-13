from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn import svm, datasets
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#
# mydata=pd.read_table("./data/S1_4_20181221.txt",sep=',',header=None)
# y=mydata.loc[:,0]
# x=mydata.loc[:,1:]
#
# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
#
# train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)#把训练集按0.2的比例划分为训练集和验证集
# #start svm
# clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)
# clf.fit(train_x,train_y)
# result  = clf.predict(test_x)#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
# #end svm ,start metrics
# test_auc = metrics.roc_auc_score(test_y,result)#验证集上的auc值
# print(test_auc)
#
# ###############################################################################################
# ###############################################################################################
# mydata1=pd.read_table("./data/S1_8_20181221.txt",sep=',',header=None)
# ######################################
# y1=mydata1.loc[:,0]
# x1=mydata1.loc[:,1:]
#
# lda = LinearDiscriminantAnalysis(n_components=10)
# lda.fit(x1,y1)
# x1 = lda.transform(x1)
#
# train_x1,test_x1,train_y1,test_y1 = train_test_split(x1,y1,test_size=0.3,random_state=0)#把训练集按0.2的比例划分为训练集和验证集
#
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=200)
#
# clf.fit(train_x1,train_y1)
# result1= clf.predict(test_x1)#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
# #end svm ,start metrics
# test_auc1 = metrics.roc_auc_score(test_y1,result1)#验证集上的auc值
# print(test_auc1)

###################################################################################

mydata=pd.read_table("./data/S2_4_20181221.txt",sep=',',header=None)
y=mydata.loc[:,0]
x=mydata.loc[:,1:]

# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)

train_x,test_x,train_y,test_y = train_test_split(x,y,test_size=0.3,random_state=0)#把训练集按0.2的比例划分为训练集和验证集
#start svm
clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)
clf.fit(train_x,train_y)
result  = clf.predict(test_x)#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
#end svm ,start metrics
test_auc = metrics.roc_auc_score(test_y,result)#验证集上的auc值
print(test_auc)

###############################################################################################
###############################################################################################
mydata1=pd.read_table("./data/S2_8_20181221.txt",sep=',',header=None)
######################################
y1=mydata1.loc[:,0]
x1=mydata1.loc[:,1:]

lda = LinearDiscriminantAnalysis(n_components=10)
lda.fit(x1,y1)
x1 = lda.transform(x1)

train_x1,test_x1,train_y1,test_y1 = train_test_split(x1,y1,test_size=0.3,random_state=0)#把训练集按0.2的比例划分为训练集和验证集

from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=200)

clf.fit(train_x1,train_y1)
result1= clf.predict(test_x1)#基于SVM对验证集做出预测，prodict_prob_y 为预测的概率
#end svm ,start metrics
test_auc1 = metrics.roc_auc_score(test_y1,result1)#验证集上的auc值
print(test_auc1)