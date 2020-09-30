from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split as ts

#使用鸢尾花数据集跑skleanr的svm模型，对鸢尾花进行分类
iris = datasets.load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = ts(X,y,test_size=0.3)

# kernel = 'rbf'
clf_rbf = svm.SVC(kernel='rbf')
clf_rbf.fit(X_train,y_train)
score_rbf = clf_rbf.score(X_test,y_test)
print("The score of rbf is : %f"%score_rbf)

# kernel = 'linear'
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train,y_train)
score_linear = clf_linear.score(X_test,y_test)
print("The score of linear is : %f"%score_linear)

# kernel = 'poly'
clf_poly = svm.SVC(kernel='poly')
clf_poly.fit(X_train,y_train)
score_poly = clf_poly.score(X_test,y_test)
print("The score of poly is : %f"%score_poly)

