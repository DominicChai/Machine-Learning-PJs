#! /usr/bin/env python 
#-*- coding: utf-8 -*-
import pandas as pd

inputfile1 = 'D:/DATABI/huawei_negative.csv' #评论汇总文件
inputfile2 = 'D:/DATABI/huawei_positive.csv'
inputfile3 = 'D:/DATABI/huaweimid.csv'
inputfile4 = 'D:/DATABI/iPhone__positive.csv'
inputfile5 = 'D:/DATABI/iPhone_negative.csv'
inputfile6 = 'D:/DATABI/iPoneXmid.csv'
inputfile7 = 'D:/DATABI/nokia__positive.csv'
inputfile8 = 'D:/DATABI/nokia_negative.csv'
inputfile9 = 'D:/DATABI/nokiamid.csv'
inputfile10 = 'D:/DATABI/vivo_negative.csv'
inputfile11 = 'D:/DATABI/vivo_positive.csv'
inputfile12 = 'D:/DATABI/vivomid.csv'

outputfile1 = 'D:/DATABI/huawei_negative.txt' #评论提取后保存路径
outputfile2 = 'D:/DATABI/huawei_positive.txt'
outputfile3 = 'D:/DATABI/huaweimid.txt'
outputfile4 = 'D:/DATABI/iPhone__positive.txt'
outputfile5 = 'D:/DATABI/iPhone_negative.txt'
outputfile6 = 'D:/DATABI/iPoneXmid.txt'
outputfile7 = 'D:/DATABI/nokia__positive.txt'
outputfile8 = 'D:/DATABI/nokia_negative.txt'
outputfile9 = 'D:/DATABI/nokiamid.txt'
outputfile10 = 'D:/DATABI/vivo_negative.txt'
outputfile11 = 'D:/DATABI/vivo_positive.txt'
outputfile12 = 'D:/DATABI/vivomid.txt'

data1 = pd.read_csv(inputfile1)#读入数据
data2 = pd.read_csv(inputfile2)
data3 = pd.read_csv(inputfile3)
data4 = pd.read_csv(inputfile4)
data5 = pd.read_csv(inputfile5)
data6 = pd.read_csv(inputfile6)
data7 = pd.read_csv(inputfile7)
data8 = pd.read_csv(inputfile8)
data9 = pd.read_csv(inputfile9)
data10 = pd.read_csv(inputfile10)
data11 = pd.read_csv(inputfile11)
data12 = pd.read_csv(inputfile12)

data1 = data1[[u'content']]#读取csv文件中content列的数据
data2 = data2[[u'content']]
data3 = data3[[u'content']]
data4 = data4[[u'content']]
data5 = data5[[u'content']]
data6 = data6[[u'content']]
data7 = data7[[u'content']]
data8 = data8[[u'content']]
data9 = data9[[u'content']]
data10 = data10[[u'content']]
data11 = data11[[u'content']]
data12 = data12[[u'content']]


data1.to_csv(outputfile1, index = False, header = False)#文本输出
data2.to_csv(outputfile2, index = False, header = False)
data3.to_csv(outputfile3, index = False, header = False)
data4.to_csv(outputfile4, index = False, header = False)
data5.to_csv(outputfile5, index = False, header = False)
data6.to_csv(outputfile6, index = False, header = False)
data7.to_csv(outputfile7, index = False, header = False)
data8.to_csv(outputfile8, index = False, header = False)
data9.to_csv(outputfile9, index = False, header = False)
data10.to_csv(outputfile10, index = False, header = False)
data11.to_csv(outputfile11, index = False, header = False)
data12.to_csv(outputfile12, index = False, header = False)
