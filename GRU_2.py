import centralizer2 as C
import numpy as np
import util
import sys
path = sys.path[0]+'/data/'
h_t=np.ones(shape=(1,16))
#模型2：将数据集划分为两方后，使用GRU模型对两方数据进行训练、在每一轮迭代中，对他们预测出来的标签值（y）再进行一次线性回归

def gru_model2(GRU_A, GRU_B, learningRate, Loopnum):
    Weight = np.random.rand(1, 2)
    # baise = np.array([[1]])
    for num in range(Loopnum):
        y_A = GRU_A.forward()
        y_B = GRU_B.forward()
        x_G=np.c_[y_A,y_B]
        print('weight1:',Weight)
        y_true=np.array(GRU_A.d).reshape(np.shape(GRU_A.d)[0],1)
        y_G=np.matmul(x_G, Weight.T)
        dw =-np.matmul((y_true - y_G).T, x_G)
        # w_gradient = -(2 / x_G.shape[0]) * np.matmul((y_true - y_G).T, x_G)
        # baise_gradient = -2 * np.matmul((y_true - y_G).T, np.ones(shape=[x_G.shape[0], 1])) / x_G.shape[0]
        Weight = Weight - learningRate * dw
        # baise = baise - learningRate * baise_gradient
        GRU_A.backprop(Weight[0][0],y_G)
        GRU_B.backprop(Weight[0][1],y_G)
        # print('GRU_A epoch : %d loss=%f'%(num,(np.math.sqrt(np.sum(np.square(GRU_A.yvalues - GRU_A.d))) / 2)))
        # print('GRU_B epoch : %d loss=%f'%(num,(np.math.sqrt(np.sum(np.square(GRU_B.yvalues - GRU_B.d))) / 2)))
        print('Total Loss:',np.math.sqrt(np.sum(np.square(y_true - y_G))) / 2)
        print('--------------------------------')
        print('weight2:',Weight)
        # print('y_g:',y_G)
    return Weight

train_xa, test_xa, train_ya, test_ya=util.load_data('dataset_high_close_volume.csv')
train_xb, test_xb, train_yb, test_yb=util.load_data('dataset_low_adj_close_volume.csv')
GRU_A=C.GRULayer(train_xa,h_t,train_ya)
GRU_B=C.GRULayer(train_xb,h_t,train_yb)
weight=gru_model2(GRU_A,GRU_B,1e-4,10000)
# with open(path+'GRU_2_A.pkl','rb') as f: #保存模型
#     GRU_A=pickle.load(f)¥
# with open(path+'GRU_2_B.pkl','rb') as f:
#     GRU_B=pickle.load(f)
# with open(path+'GRU_2_weight.pkl','rb') as f:
#     weight=pickle.load(f)
# res1=util.gru_local_predict(GRU_A,test_xa,test_ya)
# res2=util.gru_local_predict(GRU_B,test_xb,test_yb)
res1,res2=util.load_local_y()
resG=np.matmul(np.c_[res1,res2],weight.T)
util.pred_value_plt(res1,res2,resG,test_ya)
