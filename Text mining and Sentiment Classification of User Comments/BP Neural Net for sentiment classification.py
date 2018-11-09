#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:14:08 2017

@author: luogan
"""
import pandas as pd

import tensorflow as tf
#read data
df=pd.read_csv(r'huawei_cut.csv') #导入所有1000条数据
X=df.drop(['sentiment'],axis=1) #特征为x
y=df.sentiment #目标为y





from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.5,random_state=1)

from sklearn.cross_validation import train_test_split
x_train1, x_test1, y_train1, y_test1 = train_test_split(X, y,test_size=0.5,random_state=1)

from sklearn.cross_validation import train_test_split
x_train2, x_test2, y_train2, y_test2 = train_test_split(X, y,test_size=0.5,random_state=1)

from sklearn.cross_validation import train_test_split
x_train3, x_test3, y_train3, y_test3 = train_test_split(X, y,test_size=0.5,random_state=1)

from sklearn import svm  
clf1 = svm.SVC(probability=True)  # class   
clf1.fit(x_train1, y_train1)

from sklearn.naive_bayes import GaussianNB
clf2 = GaussianNB()
clf2.fit(x_train2,y_train2)

from sklearn.ensemble import RandomForestClassifier 
clf3 = RandomForestClassifier(oob_score=True, random_state=10)  
clf3.fit(x_train3,y_train3)



from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

model = Sequential() #建立模型
model.add(Dense(input_dim = 50, output_dim = 100)) #添加输入层、隐藏层的连接
model.add(Activation('tanh')) #以Relu函数为激活函数

model.add(Dense(input_dim = 100, output_dim = 100)) #添加隐藏层、隐藏层的连接
model.add(Activation('relu')) #以Relu函数为激活函数
model.add(Dropout(0.2))

model.add(Dense(input_dim = 100, output_dim = 1)) #添加隐藏层、输出层的连接
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
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

print("the accuracy is :")
print(accuracy_score(y_test, r))
from sklearn.metrics import recall_score

print("the recall is :")
print(recall_score(y_test, r))
from sklearn.metrics import confusion_matrix

print("the confusion is :")
print(confusion_matrix(y_test, r))

print("the precision is:")
print(average_precision_score(y_test, r))

print("the f-score is:")
print(f1_score(y_test, r))




print(x_test)
print(y_test)

from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt
predict = model.predict_proba(x_test.values)
predict1 = clf1.predict_proba(x_test1.values)[:,1]
print(type(predict1))
predict2 = clf2.predict_proba(x_test2.values)[:,1]
predict3 = clf3.predict_proba(x_test3.values)[:,1]
print(predict)
print(predict1)
print(predict2)
print(predict3)
fpr, tpr,_ = roc_curve(y_test,predict)
fpr1, tpr1,_ = roc_curve(y_test1,predict1)
fpr2, tpr2,_ = roc_curve(y_test2,predict2)
fpr3, tpr3,_ = roc_curve(y_test3,predict3)
roc_auc = auc(fpr,tpr)
roc_auc1 = auc(fpr1,tpr1)
roc_auc2 = auc(fpr2,tpr2)
roc_auc3 = auc(fpr3,tpr3)
plt.plot(fpr,tpr,label='Neural Network = %.2f' %roc_auc)
plt.plot(fpr1,tpr1,label='SVM = %.2f' %roc_auc1)
plt.plot(fpr2,tpr2,label='Naive Beyasian = %.2f' %roc_auc2)
plt.plot(fpr3,tpr3,label='Random forest = %.2f' %roc_auc3)

plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.legend(loc='lower right')
plt.show()
