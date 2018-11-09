# -*- coding:utf-8 -*-  


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
os.environ["PATH"] += os.pathsep + 'C:/Python 3.6/Graphviz2.38/bin/'


#在python中\是转义符，\U表示其后是UNICODE编码，因此\User这里会出错，
#在字符串前面加个r表示不进行转义就可以了

df=pd.read_csv(r'C:\Users\Dominic\Desktop\german_credit.csv') #导入所有1000条数据
x=df.drop(['Creditability'],axis=1) #特征为x
y=df.Creditability #目标为y


from sklearn.model_selection import train_test_split

#test_size 测试集的长度
#print(y_test.shape)
#print(y_test)#0 1已经交错开


#print(df) 测试导入成功

feature_names=["Account.Balance", "Duration.of.Credit..month.", "Payment.Status.of.Previous.Credit", "Purpose", "Credit.Amount",
               "Value.Savings.Stocks", "Length.of.current.employment", "Instalment.per.cent", "Sex...Marital.Status", "Guarantors",
               "Duration.in.Current.address", "Most.valuable.available.asset", "Age..years.", "Concurrent.Credits",
               "Type.of.apartment", "No.of.Credits.at.this.Bank", "Occupation", "No.of.dependents", "Telephone", "Foreign.Worker"]

#对dataframe的操作不熟悉 容易产生空的data frame
#print(feature_names)

target_names='Creditability'
#print(y)
#print(x.shape) 查看特征矩阵的行列数
#print(y.shape)查看目标向量的行列数 

from sklearn.ensemble import RandomForestClassifier #sklearn包依赖scipy包 故都需要使用pip进行加载
from sklearn.metrics import accuracy_score

def classifier(randomSeed): #随机数种子与当前系统时间有关
    x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=randomSeed)
    clf = RandomForestClassifier(n_estimators=400,max_features=20) #在利用最大投票数或平均值来预测之前，你想要建立子树的数量

#RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            #max_depth=None, max_features='auto', max_leaf_nodes=None,
            #min_samples_leaf=1, min_samples_split=2,
            #min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
            #oob_score=False, random_state=None, verbose=0,
            #warm_start=False)

    clf= clf.fit(x_train,y_train)
    result=accuracy_score(y_test,clf.predict(x_test))
    print("accuracy_score is: ")
    print(result)
    

for i in range(0,100):
    classifier(i) #进行一百次随机抽样
    
    


