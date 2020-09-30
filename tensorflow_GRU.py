import numpy as np
from tensorflow.keras.layers import  Dense, GRU
import matplotlib.pyplot as plt
from tensorflow.python.keras import Sequential
import util

#读取数据集
train_x, test_x, train_y, test_y=util.load_data2('dataset_all_420.csv')

# 训练集
print(train_x.shape)
print(train_y.shape)
print(test_x.shape)
epochs = 1000
batch_size = 5
# GRU参数: return_sequences=True GRU输出为一个序列。默认为False，输出一个值。
# input_dim： 输入单个样本特征值的维度
# input_length： 输入的时间点长度
model = Sequential()
model.add(GRU(units=10, return_sequences=True, input_dim=train_x.shape[-1], input_length=train_x.shape[1]))
model.add(GRU(units=50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=1)
y_pred = model.predict(test_x)

rms = np.sqrt(np.mean(np.power((test_y - y_pred), 2)))
print(rms)
print(y_pred.shape)
print(test_y.shape)

x_axis = np.arange(1, np.shape(test_y)[0] + 1)
plt.xlabel('Time')
plt.ylabel('Price')
plt.title('Stock Price epoch:%d batch_size:%d rms:%f'%(epochs, batch_size, rms))
plt.plot(x_axis, test_y, label='True value')
plt.plot(x_axis, y_pred.reshape(1,-1)[0], label='Predict value')
plt.legend()
plt.show()