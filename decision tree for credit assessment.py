# -*- coding:utf-8 -*-  


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
import numpy as np

os.environ["PATH"] += os.pathsep + 'C:/Python 3.6/Graphviz2.38/bin/'


#df = pd.read_csv(r'C:\Users\Dominic\Desktop\Training51.csv')
#x=df.drop(['Number','Creditability'],axis=1) #特征为x
#y=df.Creditability #目标为y
#print(y)
#在python中\是转义符，\U表示其后是UNICODE编码，因此\User这里会出错，
#在字符串前面加个r表示不进行转义就可以了

#df=pd.read_csv(r'C:\Users\Dominic\Desktop\german_credit.csv') #导入所有1000条数据
#x=df.drop(['Creditability'],axis=1) #特征为x
#y=df.Creditability #目标为y

#from sklearn.cross_validation import train_test_split
#x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

#交叉验证在一定程度上能够避免陷入局部最小值




feature_names=["Account.Balance", "Duration.of.Credit..month.", "Payment.Status.of.Previous.Credit", "Purpose", "Credit.Amount",
               "Value.Savings.Stocks", "Length.of.current.employment", "Instalment.per.cent", "Sex...Marital.Status", "Guarantors",
               "Duration.in.Current.address", "Most.valuable.available.asset", "Age..years.", "Concurrent.Credits",
               "Type.of.apartment", "No.of.Credits.at.this.Bank", "Occupation", "No.of.dependents", "Telephone", "Foreign.Worker"]



df=pd.read_csv(r'german_credit.csv') #导入所有1000条数据
x=df.drop(['Creditability'],axis=1)#特征为x

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler() #放到01之间 也可以放到正负之间
x = min_max_scaler.fit_transform(x)
print(x)

y=df.Creditability #目标为y



print(x.shape)
x=SelectKBest(mutual_info_classif,k=19).fit_transform(x,y)
#最常用的有卡方检验（χ2）。其他方法还有互信息和信息熵。
print(x.shape)




from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)

#test_size 测试集的长度
print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

#print(df) 测试导入成功

feature_names=["Account.Balance", "Duration.of.Credit..month.", "Payment.Status.of.Previous.Credit", "Purpose", "Credit.Amount",
               "Value.Savings.Stocks", "Length.of.current.employment", "Instalment.per.cent", "Sex...Marital.Status", "Guarantors",
               "Duration.in.Current.address", "Most.valuable.available.asset", "Age..years.", "Concurrent.Credits",
               "Type.of.apartment", "No.of.Credits.at.this.Bank", "Occupation", "No.of.dependents", "Telephone", "Foreign.Worker"]

#对dataframe的操作不熟悉 容易产生空的data frame
print(feature_names)


target_names='Creditability'
#print(y)
#print(x.shape) 查看特征矩阵的行列数
#print(y.shape)查看目标向量的行列数



from sklearn import tree #sklearn包依赖scipy包 故都需要使用pip进行加载
from sklearn import svm
clf = tree.DecisionTreeClassifier(criterion="entropy",max_depth=3)

#DecisionTreeClassifier(criterion=’gini’, splitter=’best’, max_depth=None, min_samples_split=2,
#min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None,
#min_impurity_decrease=0.0, min_impurity_split=None, class_weight=None, presort=False)[source]
clf= clf.fit(x_train,y_train)


#clf = svm.SVC(kernel='rbf').fit(x_train,y_train)

with open("tree.dot",'w') as f:
    f= tree.export_graphviz(clf,
                            
out_file=f,
                            
max_depth=3,
                            
impurity=True,
                            
                            class_names = ['not safe','safe'],
                            rounded=True,
                            filled=True)

import pydotplus #功能未知
from IPython.display import Image,display

dot_data = tree.export_graphviz(clf, out_file=None, 

class_names=['not safe','safe'], 
filled=True, rounded=True, 
special_characters=True)


graph = pydotplus.graph_from_dot_data(dot_data) 
Image(graph.create_png())
display(Image(graph.create_png()))



#df = pd.read_csv(r'C:\Users\Dominic\Desktop\Test50.csv')
#test_x=df.drop(['Number','Creditability'],axis=1) #特征为x
#test_y=df.Creditability #目标为y

from sklearn.metrics import accuracy_score
result=accuracy_score(y_test,clf.predict(x_test))
print("accuracy_score is: ")
print(result)


from sklearn.metrics import roc_curve,auc
import matplotlib.pyplot as plt

#predict = clf.predict(x_test)
predict = clf.predict_proba(x_test)[:,1] #获得的是决策树的概率输出
print(predict)
fpr, tpr,thresholds = roc_curve(y_test,predict)

roc_auc = auc(fpr,tpr)
plt.plot(fpr,tpr,label='area = %.2f' %roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.legend(loc='lower right')
plt.show()






