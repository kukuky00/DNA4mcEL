# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc  ###计算roc和auc
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#######################################################################################################
#   1---------------C.elegans
#######################################################################################################
#
# mydata=pd.read_table("./data/S1_4_20181221.txt",sep=',',header=None)
# y=mydata.loc[:,0]
# x=mydata.loc[:,1:]
#
# mydata1=pd.read_table("./data/S1_8_20181221.txt",sep=',',header=None)
# ######################################
# y1=mydata1.loc[:,0]
# x1=mydata1.loc[:,1:]
#
# lda = LinearDiscriminantAnalysis(n_components=10)
# lda.fit(x1,y1)
# x1 = lda.transform(x1)
#
# #####################################################################
# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = x.shape
#
# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
# ########################################################
#
# random_state = np.random.RandomState(0)
# n_samples, n_features = x1.shape
#
# # shuffle and split training and test sets
# X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=.3, random_state=0)
# ########################################################################
#
# # Learn to predict each class against the other
# clf = svm.SVC(kernel='linear', probability=True, random_state=random_state)
# #####################################################################################
# ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
# y_score = clf.fit(X_train, y_train).decision_function(X_test)
#
# # Compute ROC curve and ROC area for each class
# fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
# roc_auc = auc(fpr, tpr)  ###计算auc的值
#
# #################################################################
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=200)
# y_score1 =clf.fit(X_train1, y_train1).decision_function(X_test1)
#
# # Compute ROC curve and ROC area for each class
# fpr1, tpr1, threshold1 = roc_curve(y_test1, y_score1)  ###计算真正率和假正率
# roc_auc1 = auc(fpr1, tpr1)  ###计算auc的值
# ########################################################################################
#
# plt.figure()
# lw = 2
# plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange',lw=lw, label='iDNA4mC (area = %0.2f)' % roc_auc) ##假正率为横坐标，真正率为纵坐标做曲线
# plt.plot(fpr1, tpr1, color='green',lw=lw, label='DNA4mCEL (area = %0.2f)' % roc_auc1)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve of C.elegans')
# plt.legend(loc="lower right")
# plt.savefig("S1_ROC.png")
# plt.show()

#######################################################################################################
#  2---------------D. melanogaster
#######################################################################################################
#
# mydata=pd.read_table("./data/S2_4_20181221.txt",sep=',',header=None)
# y=mydata.loc[:,0]
# x=mydata.loc[:,1:]
#
# mydata1=pd.read_table("./data/S2_8_20181221.txt",sep=',',header=None)
# ######################################
# y1=mydata1.loc[:,0]
# x1=mydata1.loc[:,1:]
#
# lda = LinearDiscriminantAnalysis(n_components=10)
# lda.fit(x1,y1)
# x1 = lda.transform(x1)
#
# #####################################################################
# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = x.shape
#
# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
# ########################################################
#
# random_state = np.random.RandomState(0)
# n_samples, n_features = x1.shape
#
# # shuffle and split training and test sets
# X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=.3, random_state=0)
# ########################################################################
#
# # Learn to predict each class against the other
# svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
# #####################################################################################
# ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
# y_score = svm.fit(X_train, y_train).decision_function(X_test)
#
# # Compute ROC curve and ROC area for each class
# fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
# roc_auc = auc(fpr, tpr)  ###计算auc的值
#
# #################################################################
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=200)
# y_score1 =clf.fit(X_train1, y_train1).decision_function(X_test1)
#
# # Compute ROC curve and ROC area for each class
# fpr1, tpr1, threshold1 = roc_curve(y_test1, y_score1)  ###计算真正率和假正率
# roc_auc1 = auc(fpr1, tpr1)  ###计算auc的值
# ########################################################################################
#
# plt.figure()
# lw = 2
# plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange',lw=lw, label='iDNA4mC (area = %0.2f)' % roc_auc) ##假正率为横坐标，真正率为纵坐标做曲线
# plt.plot(fpr1, tpr1, color='green',lw=lw, label='DNA4mCEL (area = %0.2f)' % roc_auc1)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve of D. melanogaster')
# plt.legend(loc="lower right")
# plt.savefig("S2_ROC.png")
# plt.show()


#######################################################################################################
#  3---------------A.thaliana
# #######################################################################################################
#
# mydata=pd.read_table("./data/S3_1_20181221.txt",sep=',',header=None)
# y=mydata.loc[:,0]
# x=mydata.loc[:,1:]
#
# mydata1=pd.read_table("./data/S3_8_20181221.txt",sep=',',header=None)
# ####################################################################
# y1=mydata1.loc[:,0]
# x1=mydata1.loc[:,1:]
#
# lda = LinearDiscriminantAnalysis(n_components=10)
# lda.fit(x1,y1)
# x1 = lda.transform(x1)
#
# #####################################################################
# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = x.shape
#
# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
# ################################################################################
#
# random_state = np.random.RandomState(0)
# n_samples, n_features = x1.shape
#
# # shuffle and split training and test sets
# X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=.3, random_state=0)
# ########################################################################
#
# # Learn to predict each class against the other
# svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
# #####################################################################################
# ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
# y_score = svm.fit(X_train, y_train).decision_function(X_test)
#
# # Compute ROC curve and ROC area for each class
# fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
# roc_auc = auc(fpr, tpr)  ###计算auc的值
#
# #################################################################
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=200)
# y_score1 =clf.fit(X_train1, y_train1).decision_function(X_test1)
#
# # Compute ROC curve and ROC area for each class
# fpr1, tpr1, threshold1 = roc_curve(y_test1, y_score1)  ###计算真正率和假正率
# roc_auc1 = auc(fpr1, tpr1)  ###计算auc的值
# ########################################################################################
#
# plt.figure()
# lw = 2
# plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange',lw=lw, label='iDNA4mC (area = %0.2f)' % roc_auc) ##假正率为横坐标，真正率为纵坐标做曲线
# plt.plot(fpr1, tpr1, color='green',lw=lw, label='DNA4mCEL (area = %0.2f)' % roc_auc1)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve of A.thaliana')
# plt.legend(loc="lower right")
# plt.savefig("S3_ROC.png")
# plt.show()


#######################################################################################################
#  4---------------E.coli
#######################################################################################################
# mydata=pd.read_table("./data/S4_1_20181221.txt",sep=',',header=None)
# y=mydata.loc[:,0]
# x=mydata.loc[:,1:]
#
# from sklearn.preprocessing import MinMaxScaler
# scaler1 = MinMaxScaler(feature_range=(0, 1))
# x = scaler1.fit_transform(x)
#
# mydata1=pd.read_table("./data/S4_9_20181221.txt",sep=',',header=None)
# ####################################################################
# y1=mydata1.loc[:,0]
# x1=mydata1.loc[:,1:]
#
# #####################################################################
# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = x.shape
#
# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
# ################################################################################
#
# random_state = np.random.RandomState(0)
# n_samples, n_features = x1.shape
#
# # shuffle and split training and test sets
# X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=.3, random_state=0)
# ########################################################################
#
# # Learn to predict each class against the other
# svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
# #####################################################################################
# ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
# y_score = svm.fit(X_train, y_train).decision_function(X_test)
#
# # Compute ROC curve and ROC area for each class
# fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
# roc_auc = auc(fpr, tpr)  ###计算auc的值
#
# #################################################################
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=200)
# y_score1 =clf.fit(X_train1, y_train1).decision_function(X_test1)
#
# # Compute ROC curve and ROC area for each class
# fpr1, tpr1, threshold1 = roc_curve(y_test1, y_score1)  ###计算真正率和假正率
# roc_auc1 = auc(fpr1, tpr1)  ###计算auc的值
# ########################################################################################
#
# plt.figure()
# lw = 2
# plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange',lw=lw, label='iDNA4mC (area = %0.2f)' % roc_auc) ##假正率为横坐标，真正率为纵坐标做曲线
# plt.plot(fpr1, tpr1, color='green',lw=lw, label='DNA4mCEL (area = %0.2f)' % roc_auc1)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve of E.coli')
# plt.legend(loc="lower right")
# plt.savefig("S4_ROC.png")
# plt.show()

#
# #######################################################################################################
# #  5--------------G.subterraneus
# #######################################################################################################
# mydata=pd.read_table("./data/S5_6_20181221.txt",sep=',',header=None)
# y=mydata.loc[:,0]
# x=mydata.loc[:,1:]
#
# from sklearn.preprocessing import Normalizer
# scaler2 = Normalizer().fit(x)
# x = scaler2.transform(x)
#
#
# mydata1=pd.read_table("./data/S5_8_20181221.txt",sep=',',header=None)
# ####################################################################
# y1=mydata1.loc[:,0]
# x1=mydata1.loc[:,1:]
#
# # ####lda : lower the dimension ####
# lda = LinearDiscriminantAnalysis(n_components=10)
# lda.fit(x1,y1)
# x1 = lda.transform(x1)
# #####################################################################
# # Add noisy features to make the problem harder
# random_state = np.random.RandomState(0)
# n_samples, n_features = x.shape
#
# # shuffle and split training and test sets
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
# ################################################################################
#
# random_state = np.random.RandomState(0)
# n_samples, n_features = x1.shape
#
# # shuffle and split training and test sets
# X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=.3, random_state=0)
# ########################################################################
#
# # Learn to predict each class against the other
# svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
# #####################################################################################
# ###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
# y_score = svm.fit(X_train, y_train).decision_function(X_test)
#
# # Compute ROC curve and ROC area for each class
# fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
# roc_auc = auc(fpr, tpr)  ###计算auc的值
#
# #################################################################
# from sklearn.ensemble import GradientBoostingClassifier
# clf = GradientBoostingClassifier(n_estimators=200)
# y_score1 =clf.fit(X_train1, y_train1).decision_function(X_test1)
#
# # Compute ROC curve and ROC area for each class
# fpr1, tpr1, threshold1 = roc_curve(y_test1, y_score1)  ###计算真正率和假正率
# roc_auc1 = auc(fpr1, tpr1)  ###计算auc的值
# ########################################################################################
#
# plt.figure()
# lw = 2
# plt.figure(figsize=(10, 10))
# plt.plot(fpr, tpr, color='darkorange',lw=lw, label='iDNA4mC (area = %0.2f)' % roc_auc) ##假正率为横坐标，真正率为纵坐标做曲线
# plt.plot(fpr1, tpr1, color='green',lw=lw, label='DNA4mCEL (area = %0.2f)' % roc_auc1)
# plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC curve of G.subterraneus')
# plt.legend(loc="lower right")
# plt.savefig("S5_ROC.png")
# plt.show()


#######################################################################################################
#  6--------------G.pickeringii
#######################################################################################################
mydata=pd.read_table("./data/S6_6_20181221.txt",sep=',',header=None)
y=mydata.loc[:,0]
x=mydata.loc[:,1:]
from sklearn.preprocessing import Normalizer
scaler2 = Normalizer().fit(x)
x = scaler2.transform(x)

mydata1=pd.read_table("./data/S6_8_20181221.txt",sep=',',header=None)
####################################################################
y1=mydata1.loc[:,0]
x1=mydata1.loc[:,1:]

#####################################################################
# Add noisy features to make the problem harder
random_state = np.random.RandomState(0)
n_samples, n_features = x.shape

# shuffle and split training and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=.3, random_state=0)
################################################################################

random_state = np.random.RandomState(0)
n_samples, n_features = x1.shape

# shuffle and split training and test sets
X_train1, X_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size=.3, random_state=0)
########################################################################

# Learn to predict each class against the other
svm = svm.SVC(kernel='linear', probability=True, random_state=random_state)
#####################################################################################
###通过decision_function()计算得到的y_score的值，用在roc_curve()函数中
y_score = svm.fit(X_train, y_train).decision_function(X_test)

# Compute ROC curve and ROC area for each class
fpr, tpr, threshold = roc_curve(y_test, y_score)  ###计算真正率和假正率
roc_auc = auc(fpr, tpr)  ###计算auc的值

#################################################################
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(n_estimators=200)
y_score1 =clf.fit(X_train1, y_train1).decision_function(X_test1)

# Compute ROC curve and ROC area for each class
fpr1, tpr1, threshold1 = roc_curve(y_test1, y_score1)  ###计算真正率和假正率
roc_auc1 = auc(fpr1, tpr1)  ###计算auc的值
########################################################################################

plt.figure()
lw = 2
plt.figure(figsize=(10, 10))
plt.plot(fpr, tpr, color='darkorange',lw=lw, label='iDNA4mC (area = %0.2f)' % roc_auc) ##假正率为横坐标，真正率为纵坐标做曲线
plt.plot(fpr1, tpr1, color='green',lw=lw, label='DNA4mCEL (area = %0.2f)' % roc_auc1)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve of G.pickeringii')
plt.legend(loc="lower right")
plt.savefig("S6_ROC.png")
plt.show()