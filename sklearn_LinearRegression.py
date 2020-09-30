import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
path = sys.path[0] + '/data/'

#使用sklearn的线性回归模型对股票进行预测

def spilt_data(data):
    # data=data.drop(columns=['date','ticker','adj_close'])
    # y=data.pop('close')
    data=data.drop(columns=['Date'])
    y=data.pop('Open')
    train_x=data[:1000]
    test_x=data[1000:]
    train_y=y[:1000]
    test_y=y[1000:]
    # train_x, test_x, train_y, test_y = train_test_split(data, y, test_size=0.15,random_state=0)
    return  train_x, test_x, train_y, test_y

data=pd.read_csv(path+'dataset_all.csv')
train_x, test_x, train_y, test_y=spilt_data(data)
train_x=train_x.astype('float64')
train_y=train_y.astype('float64')

#LR
model = linear_model.LinearRegression()
model.fit(train_x,train_y)
print(model.coef_)
y = model.predict(test_x)

x_axis = np.arange(1, np.shape(test_x)[0] + 1)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price')
plt.plot(x_axis, y, label='Predict value')
plt.plot(x_axis, test_y, label='True value')
plt.legend()
plt.show()
