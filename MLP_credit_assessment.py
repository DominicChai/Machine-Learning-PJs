#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:14:08 2017

@author: luogan
"""
import pandas as pd

import tensorflow as tf
#read data
df=pd.read_csv(r'german_credit.csv') #导入所有1000条数据
X=df.drop(['Creditability'],axis=1) #特征为x
y=df.Creditability #目标为y


#change categorical

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
d = defaultdict(LabelEncoder)
X_trans = X.apply(lambda x: d[x.name].fit_transform(x))
X_trans.head()

#X_trans.to_excel('X_trans.xls') 
##############
data_train=X_trans
data_max = data_train.max()
data_min = data_train.min()
data_mean = data_train.mean()
#
# data_std = data_train.std()
X_train1 = (data_train-data_max)/(data_max-data_min)


#y=0.5*(y+1)
#print(y)

#random take train and test

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X_train1, y,test_size=0.1,random_state=1)



#x_train.to_excel('xx_trans.xls') 

#y_train.to_excel('y_trans.xls') 




#call decision tree
#from sklearn import tree
#clf = tree.DecisionTreeClassifier(max_depth=10)
#clf = clf.fit(X_train, y_train)

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

model = Sequential() #建立模型
model.add(Dense(input_dim = 20, output_dim = 48)) #添加输入层、隐藏层的连接
model.add(Activation('tanh')) #以Relu函数为激活函数

model.add(Dense(input_dim = 48, output_dim = 48)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dropout(0.2))

model.add(Dense(input_dim = 48, output_dim = 36)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dropout(0.2))
model.add(Dense(input_dim = 36, output_dim = 36)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数

model.add(Dense(input_dim = 36, output_dim = 12)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dense(input_dim = 12, output_dim = 12)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数


model.add(Dense(input_dim = 12, output_dim = 1)) #添加隐藏层、输出层的连接
model.add(Activation('sigmoid')) #以sigmoid函数为激活函数

#编译模型，损失函数为binary_crossentropy，用adam法求解
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(x_train.values, y_train.values, nb_epoch = 200, batch_size = 2000) #训练模型


#nb_epoch=5000 0.748

#nb_epoch=1000 0.724


#nb_epoch=100 0.784


#nb_epoch=200 0.768



r = pd.DataFrame(model.predict_classes(x_test.values))
'''
r = pd.DataFrame(model.predict(x_test.values))
rr=r.values
tr=rr.flatten()

for i in range(tr.shape[0]):
    if tr[i]>0.5:
        tr[i]=1
    else:

        tr[i]=0
'''        
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, r)) 



from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
predict = model.predict_classes(x_test.values)
fpr, tpr,_ = roc_curve(y_test,predict)
roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.legend(loc='lower right')
plt.show()
