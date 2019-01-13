import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.datasets import load_iris

from sklearn import svm

iris = load_iris()
#这里是按照6:4对训练集测试集进行划分
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=.4, random_state=0)
######################################################################
# clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
# print(clf.score(X_test, y_test))
#####################################cross validation=5 #################################
from sklearn.model_selection import cross_val_score
# clf = svm.SVC(kernel='linear', C=1)
# # scores = cross_val_score(clf, iris.data, iris.target, cv=5)
# # print(scores)
#
from sklearn.model_selection import ShuffleSplit
# n_samples = iris.data.shape[0]
cv = ShuffleSplit(n_splits=4, test_size=.3, random_state=0)
# print(cross_val_score(clf, iris.data, iris.target, cv=cv))


from sklearn import preprocessing
from sklearn.pipeline import make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
# print(cross_val_score(clf, iris.data, iris.target, cv=cv))

# from sklearn.model_selection import cross_val_predict
# from sklearn import metrics
# predicted = cross_val_predict(clf, iris.data, iris.target, cv=10)
# # print(predicted)
# print(metrics.accuracy_score(iris.target, predicted))

from sklearn.model_selection import KFold


############################################################################################
# from sklearn import datasets	#自带数据集
# from sklearn.model_selection import train_test_split,cross_val_score	#划分数据 交叉验证
# from sklearn.neighbors import KNeighborsClassifier  #一个简单的模型，只有K一个参数，类似K-means
# import matplotlib.pyplot as plt
# iris = datasets.load_iris()		#加载sklearn自带的数据集
# X = iris.data 			#这是数据
# y = iris.target 		#这是每个数据所对应的标签
# train_X,test_X,train_y,test_y = train_test_split(X,y,test_size=1/3,random_state=3)	#这里划分数据以1/3的来划分 训练集训练结果 测试集测试结果
# k_range = range(1,31)
# cv_scores = []		#用来放每个模型的结果值
# for n in k_range:
#     knn = KNeighborsClassifier(n)   #knn模型，这里一个超参数可以做预测，当多个超参数时需要使用另一种方法GridSearchCV
#     scores = cross_val_score(knn,train_X,train_y,cv=10,scoring='accuracy')  #cv：选择每次测试折数  accuracy：评价指标是准确度,可以省略使用默认值
#     cv_scores.append(scores.mean())
# plt.plot(k_range,cv_scores)
# plt.xlabel('K')
# plt.ylabel('Accuracy')		#通过图像选择最好的参数
# plt.show()
# best_knn = KNeighborsClassifier(n_neighbors=3)	# 选择最优的K=3传入模型
# best_knn.fit(train_X,train_y)			#训练模型
# print(best_knn.score(test_X,test_y))	#看看评分
