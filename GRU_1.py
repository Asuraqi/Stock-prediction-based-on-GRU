import pickle
import centralizer as C
import numpy as np
import util
import sys
path = sys.path[0] + '/data/'
h_t=np.ones(shape=(1,16))
#模型1：将数据集划分为两方后，使用GRU模型对两方数据进行训练、预测，对他们预测出来的标签值（y）再进行一次线性回归

def gru_model1(data_x, data_y, learningRate, Loopnum):
    Weight = np.ones(shape=(1, data_x.shape[1]))/2
    baise = np.array([[1]])
    data_y=np.array(data_y).reshape(np.shape(data_x)[0],1)
    print(np.shape(Weight))
    for num in range(Loopnum):
        y_pred = np.matmul(data_x, Weight.T)+baise
        w_gradient = -(2 / data_x.shape[0])* np.matmul((data_y - y_pred).T, data_x)
        baise_gradient = -2 * np.matmul((data_y - y_pred).T, np.ones(shape=[data_x.shape[0], 1])) / data_x.shape[0]
        Weight =Weight - learningRate * w_gradient
        baise = baise - learningRate * baise_gradient
        print('epoch : %d Total Loss: %f'%(num,np.math.sqrt(np.sum(np.square(data_y - y_pred))) / 2))
        print('-----------------------------------')
    return (Weight, baise)

train_xa, test_xa, train_ya, test_ya=util.load_data('dataset_high_close_volume.csv')
train_xb, test_xb, train_yb, test_yb=util.load_data('dataset_low_adj_close_volume.csv')
GRU_A=C.GRULayer(train_xa, h_t, train_ya)
GRU_B=C.GRULayer(train_xb, h_t, train_yb)
y_A=util.gru_local_train(GRU_A, 10000)
y_B=util.gru_local_train(GRU_B, 10000)
# with open(path+'GRU_1_A.pkl','rb') as f:    #保存模型
#     gru_a=pickle.load(f)
# with open(path+'GRU_1_B.pkl','rb') as f:
#     gru_b=pickle.load(f)

# y_A=gru_a.yvalues
# y_B=gru_b.yvalues
print(y_A)
y_G_train=np.c_[y_A,y_B]
print('y_local_pred:',y_G_train)
y_G_true=train_ya
weight,baise=gru_model1(y_G_train,y_G_true,1e-3,70000)
print('weight',weight)
with open(path+'GRU_1_A.pkl','wb') as f:
    pickle.dump(GRU_A,f)
with open(path+'GRU_1_B.pkl','wb') as g:
    pickle.dump(GRU_B,g)
res1=util.gru_local_predict(GRU_A,test_xa,test_ya)
res2=util.gru_local_predict(GRU_B,test_xb,test_yb)
resG=np.matmul(np.c_[res1,res2], weight.T)+baise
util.pred_value_plt(res1,res2,resG,test_yb)


