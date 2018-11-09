#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import pandas as pd
import openpyxl
import tensorflow as tf
#read data
df = pd.read_excel(r'..\fsi-2006~2017.xlsx')
#print(df)
X=df.drop(['Year','Rank','Total','Country','Mark_from_Entrophy'],axis=1) #
#X for input
y=df.Mark_from_Entrophy 
#Y for output
from sklearn.cross_validation import train_test_split
#split dataset into training set/validation set/test set
x_train, x_test, y_train, y_test = train_test_split(X, y,test_size=0.1,random_state=1)
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
model = Sequential()#initialization
model.add(Dense(input_dim = 12, output_dim = 12))
#add connection between layer
model.add(Activation('relu')) #use ReLU as activation function
model.add(Dense(input_dim = 12, output_dim = 1))
#add connection between layers
model.add(Activation('relu'))

#compile model,and use "mean_sqaured_error"as loss function for regression
#use adam method as optimizer
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train.values, y_train.values, nb_epoch = 15, batch_size = 20)
#train NN
r = pd.DataFrame(model.predict(x_test.values))
writer = pd.ExcelWriter('output.xlsx')
#result = pd.concat([x_test, r], axis=1)
x_test.to_excel(writer,'Sheet1')
r.to_excel(writer,'Sheet2')
writer.save()



