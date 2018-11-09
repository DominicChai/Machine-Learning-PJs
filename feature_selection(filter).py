import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

from sklearn.feature_selection import mutual_info_classif
import numpy as np


feature_names=["Account.Balance", "Duration.of.Credit..month.", "Payment.Status.of.Previous.Credit", "Purpose", "Credit.Amount",
               "Value.Savings.Stocks", "Length.of.current.employment", "Instalment.per.cent", "Sex...Marital.Status", "Guarantors",
               "Duration.in.Current.address", "Most.valuable.available.asset", "Age..years.", "Concurrent.Credits",
               "Type.of.apartment", "No.of.Credits.at.this.Bank", "Occupation", "No.of.dependents", "Telephone", "Foreign.Worker"]

df=pd.read_csv(r'C:\Users\Dominic\Desktop\german_credit.csv') #导入所有1000条数据
x=df.drop(['Creditability'],axis=1)#特征为x
print(x)
y=df.Creditability #目标为y


#消除量纲
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler() #放到01之间 也可以放到正负之间
x = min_max_scaler.fit_transform(x)

print(sum(x))
#结论：消除量纲之前 卡方和互信息准则选出来的特征不同 消除之后 就相同了



print(x.shape)
x_new=SelectKBest(mutual_info_classif,k=5).fit_transform(x,y)
#print(type(x_new)) 查看数据结构类型 非常有用


#x_new.tolist()
print(x_new.shape)
print(sum(x_new))

print(x.shape)

for index in range(0,len(sum(x_new))):
    for num in range(0,len(sum(x))):
        if(sum(x)[num]==sum(x_new)[index]):
            print(feature_names[num]) #显示筛选出来的特征变量

#结论：不同特征选择准则选出来的特征组合也不同 即使消除了量纲

#x_new=SelectKBest(chi2,k=1).fit_transform(x,y)
#print(x_new.shape)
#print(sum(x_new))


#print(feature_names[0])
#print(sum(x_new))
#print(sum(x_new)[0])
#print(sum(x))
#print(len(sum(x)))
