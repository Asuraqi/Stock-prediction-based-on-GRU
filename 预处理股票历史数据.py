import pandas as pd
import sys
path = sys.path[0]+'/data/'
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
#对股票源数据预处理 股票源数据获取网址:https://cn.investing.com/equities
def getvolume(data):
    if data.rfind('M')>-1:
        return float(data.split('M')[0])*1000
    else:
        return float(data.split('K')[0])

def encode(data):
    data['ticker'] = LabelEncoder().fit_transform(data['ticker'])
    return data

def minmax_scaler(data):
    features=['收盘','开盘','高','低','交易量']
    for feature in features:
        data[feature] = preprocessing.minmax_scale(data[feature])
        # data[feature]=scaler.fit_transform(data[feature].values.reshape(-1,1))
    return data

data=pd.read_csv('data/AMZN.csv',encoding = 'gb2312')
data['交易量']=data['交易量'].apply(lambda x:float(getvolume(x)))
print(data['交易量'])
data=data.head(420)
data=minmax_scaler(data)
data=data.drop(columns=['涨跌幅'])
data=data.iloc[::-1]
print('转化后',data)
data.to_csv('data/AMZN-处理后.csv')
