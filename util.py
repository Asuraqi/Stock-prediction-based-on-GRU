import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
path = sys.path[0]+'/data/'
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

#一些可以通用的工具包，比如划分数据集、模型训练、预测、画图、数据预处理等
def load_data(dataset):#正常形式获取数据
    data=pd.read_csv(path+dataset)
    data = data.drop(columns=['Date'])
    y = data.pop('Open')
    # data=np.array(y).reshape(np.shape(y)[0],1)
    train_x=data[:400]
    test_x=data[400:]
    train_y=y[:400]
    test_y=y[400:]
    # train_x, test_x, train_y, test_y=train_test_split(data, y, test_size=0.15)
    train_x=np.array(train_x)
    test_x=np.array(test_x)
    train_y=np.array(train_y)
    test_y=np.array(test_y)
    return  train_x, test_x, train_y, test_y

def load_data2(dataset):#分步长的形式获取数据
    data = pd.read_csv(path + dataset)
    data = data.drop(columns=['Date'])
    y = data.pop('Open')
    # data=np.array(y).reshape(np.shape(y)[0],1)
    train_x = data[:400]
    test_x = data[400:]
    train_y = y[:400]
    test_y = y[400:]
    # train_x, test_x, train_y, test_y=train_test_split(data, y, test_size=0.15)
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    x_train = []
    y_train = []

    x_test = []
    y_test = []
    for i in range(10, np.shape(train_x)[0]):
        x_train.append(train_x[i - 10:i, ])
        y_train.append(train_y[i, ])
    x_train, y_train = np.array(x_train), np.array(y_train)
    # x_train = np.reshape(np.shape(train_x)[0], (x_train.shape[0], 10, 1))
    for i in range(10, np.shape(test_x)[0]):
        x_test.append(test_x[i - 10:i, ])
        y_test.append(test_y[i, ])
    x_test, y_test = np.array(x_test), np.array(y_test)
    # x_test = np.reshape(x_test, (x_test.shape[0], 10, 1))
    return x_train, x_test, y_train, y_test

def gru_local_train(GRU,nepoch):#模型训练
    for epoch in range(nepoch):
        GRU.forward()
        GRU.backprop()
        print('epoch : %d loss=%f'%(epoch,(np.math.sqrt(np.sum(np.square(GRU.yvalues - GRU.d))) / 2)))
    return GRU.yvalues

def gru_local_predict(GRU, test_x,test_y):#模型预测
    GRU.change_input(test_x,test_y)
    print(GRU.hvalues[np.shape(GRU.x_values)[0]-1])
    h=GRU.forward2(GRU.hvalues[np.shape(GRU.x_values)[0]-1])
    print(h[0, :])
    loss = np.sum(np.square(GRU.yvalues - GRU.d))
    print("Total Loss :%f" % (np.math.sqrt(loss) / 2))
    return GRU.yvalues

def load_local_y():#获取两方预测的y值
    with open(path+'GRU_1_A.pkl','rb') as f:
         gru_a = pickle.load(f)
    with open(path + 'GRU_1_B.pkl', 'rb') as f:
         gru_b = pickle.load(f)
    train_xa, test_xa, train_ya, test_ya = load_data('dataset_high_close_volume.csv')
    train_xb, test_xb, train_yb, test_yb = load_data('dataset_low_adj_close_volume.csv')
    res1 = gru_local_predict(gru_a, test_xa, test_ya)
    res2 = gru_local_predict(gru_b, test_xb, test_yb)
    return res1,res2

def pred_value_plt(local_A,local_B,local_G,truevalue):#画图
    x_axis = np.arange(1, np.shape(local_A)[0]+1)
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.title('Stock Price')
    plt.plot(x_axis, local_A, label='Local Prediction 1')
    plt.plot(x_axis, local_B, label='Local Prediction 2')
    plt.plot(x_axis,local_G,label='Overall Prediction')
    plt.plot(x_axis, truevalue, label='True value')
    plt.legend()
    plt.show()

def encode(data):#one-hot编码
    data['ticker'] = LabelEncoder().fit_transform(data['ticker'])
    return data

def minmax_scaler(data):#数据归一化
    id_features=['Open','Close','Adj Close','Low','High','Volume']
    for feature in id_features:
        data[feature] = preprocessing.minmax_scale(data[feature])
        # data[feature]=scaler.fit_transform(data[feature].values.reshape(-1,1))
    return data

def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res

data=pd.read_csv(path+'dataset_all.csv')
# data=data[data['ticker']=='AHH']
# data=encode(data)
# data=minmax_scaler(data)
data=data.head(420)
# data['mid']=(data['High']+data['Low'])/2
data.to_csv(path+'dataset_all_420.csv',index=False)