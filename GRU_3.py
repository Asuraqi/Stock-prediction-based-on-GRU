import centralizer3 as C
import numpy as np
import util
h_t=np.ones(shape=(1,16))
#模型1：将数据集划分为两方后，使用GRU模型对两方数据进行训练、在每一轮迭代中，对他们得到的ht，再进行一次线性回归

def sigmoid(x):
    res = 1 / (1 + np.exp(-x))
    return res
def diffsigmoid(x):
    res = x*(1-x)
    return res
def gru_model3(GRU_A, GRU_B, testxa,testxb,testya,testyb,learningRate, Loopnum):
    W_hy_ = np.random.rand(GRU_A.hdim+GRU_B.hdim, GRU_A.ydim) * 2 - 1
    dWhy = np.zeros(shape=(GRU_A.hdim+GRU_B.hdim, GRU_A.ydim))
    for num in range(Loopnum):
        h_A = GRU_A.forward()
        h_B = GRU_B.forward()
        h_G = np.c_[h_A,h_B]
        y_G=sigmoid(np.matmul(h_G,W_hy_))
        y_true=np.array(GRU_A.d).reshape(np.shape(GRU_A.x_values)[0],1)
        for t in range(GRU_A.uNum):
            delta_y = (y_G[t]-y_true[t])*diffsigmoid(y_G[t])/GRU_A.uNum
            dWhy = dWhy + (h_G[t]*delta_y).reshape(np.size(dWhy),1)
            W_hy_ -= learningRate * dWhy
        print('GRU_g epoch : %d loss=%f' % (num, (np.math.sqrt(np.sum(np.square(y_G - GRU_B.d))) / (2*GRU_A.uNum))))
        GRU_A.backprop(y_G,W_hy_[:GRU_A.hdim])
        GRU_B.backprop(y_G,W_hy_[GRU_A.hdim:])
    GRU_A.change_input(testxa, testya)
    ha=GRU_A.forward2(h_A[GRU_A.uNum-1])
    GRU_B.change_input(testxb, testyb)
    hb=GRU_B.forward2(h_B[GRU_B.uNum-1])
    h_g = np.c_[ha,hb]
    y_g=np.zeros(shape=np.shape(testya))
    for t in range(np.shape(testxa)[0]):
        y_g[t] = sigmoid(np.matmul(h_g[t, :], W_hy_))
    print(y_g)
    return y_g

train_xa, test_xa, train_ya, test_ya=util.load_data('dataset_high_close_volume.csv')
train_xb, test_xb, train_yb, test_yb=util.load_data('dataset_low_adj_close_volume.csv')
GRU_A=C.GRULayer(train_xa,h_t,train_ya)
GRU_B=C.GRULayer(train_xb,h_t,train_yb)
resG=gru_model3(GRU_A,GRU_B,test_xa,test_xb,test_ya,test_yb,1e-8,1000)
res1,res2=util.load_local_y()
util.pred_value_plt(res1,res2,resG,test_yb)