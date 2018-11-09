#-*- coding: utf-8 -*-
import pandas as pd
import jieba #导入结巴分词，需要自行下载安装

#参数初始化
inputfile1 = 'D:/DATABI/huawei_negative.txt' 
inputfile2 = 'D:/DATABI/huawei_positive.txt'
inputfile3 = 'D:/DATABI/huaweimid.txt'
inputfile4 = 'D:/DATABI/iPhone__positive.txt'
inputfile5 = 'D:/DATABI/iPhone_negative.txt'
inputfile6 = 'D:/DATABI/iPoneXmid.txt'
inputfile7 = 'D:/DATABI/nokia__positive.txt'
inputfile8 = 'D:/DATABI/nokia_negative.txt'
inputfile9 = 'D:/DATABI/nokiamid.txt'
inputfile10 = 'D:/DATABI/vivo_negative.txt'
inputfile11 = 'D:/DATABI/vivo_positive.txt'
inputfile12 = 'D:/DATABI/vivomid.txt'

outputfile1 = 'D:/DATABI/huawei_negativecut.txt' 
outputfile2 = 'D:/DATABI/huawei_positivecut.txt'
outputfile3 = 'D:/DATABI/huaweimidcut.txt'
outputfile4 = 'D:/DATABI/iPhone__positivecut.txt'
outputfile5 = 'D:/DATABI/iPhone_negativecut.txt'
outputfile6 = 'D:/DATABI/iPoneXmidcut.txt'
outputfile7 = 'D:/DATABI/nokia__positivecut.txt'
outputfile8 = 'D:/DATABI/nokia_negativecut.txt'
outputfile9 = 'D:/DATABI/nokiamidcut.txt'
outputfile10 = 'D:/DATABI/vivo_negativecut.txt'
outputfile11 = 'D:/DATABI/vivo_positivecut.txt'
outputfile12 = 'D:/DATABI/vivomidcut.txt'

data1 = pd.read_csv(inputfile1,header = None) #读入数据
data2 = pd.read_csv(inputfile2,header = None)
data3 = pd.read_csv(inputfile3,header = None)
data4 = pd.read_csv(inputfile4,header = None)
data5 = pd.read_csv(inputfile5,header = None)
data6 = pd.read_csv(inputfile6,header = None)
data7 = pd.read_csv(inputfile7,header = None)
data8 = pd.read_csv(inputfile8,header = None)
data9 = pd.read_csv(inputfile9,header = None)
data10 = pd.read_csv(inputfile10,header = None)
data11 = pd.read_csv(inputfile11,header = None)
data12 = pd.read_csv(inputfile12,header = None)


mycut = lambda s: ' '.join(jieba.cut(s)) #自定义简单分词函数
data1 = data1[0].apply(mycut) #通过“广播”形式分词，加快速度。
data2 = data2[0].apply(mycut)
data3 = data3[0].apply(mycut)
data4 = data4[0].apply(mycut)
data5 = data5[0].apply(mycut)
data6 = data6[0].apply(mycut)
data7 = data7[0].apply(mycut)
data8 = data8[0].apply(mycut)
data9 = data9[0].apply(mycut)
data10 = data10[0].apply(mycut)
data11 = data11[0].apply(mycut)
data12 = data12[0].apply(mycut)

data1.to_csv(outputfile1, index = False, header = False, encoding = 'utf-8') #保存结果
data2.to_csv(outputfile2, index = False, header = False, encoding = 'utf-8')
data3.to_csv(outputfile3, index = False, header = False, encoding = 'utf-8')
data4.to_csv(outputfile4, index = False, header = False, encoding = 'utf-8')
data5.to_csv(outputfile5, index = False, header = False, encoding = 'utf-8')
data6.to_csv(outputfile6, index = False, header = False, encoding = 'utf-8')
data7.to_csv(outputfile7, index = False, header = False, encoding = 'utf-8')
data8.to_csv(outputfile8, index = False, header = False, encoding = 'utf-8')
data9.to_csv(outputfile9, index = False, header = False, encoding = 'utf-8')
data10.to_csv(outputfile10, index = False, header = False, encoding = 'utf-8')
data11.to_csv(outputfile11, index = False, header = False, encoding = 'utf-8')
data12.to_csv(outputfile12, index = False, header = False, encoding = 'utf-8')
