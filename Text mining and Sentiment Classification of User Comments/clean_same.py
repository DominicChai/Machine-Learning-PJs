#-*- coding: utf-8 -*-
import pandas as pd

inputfile1 = 'D:/DATABI/huawei_negative.txt' #评论文件
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

outputfile1 = 'D:/DATABI/huawei_negativecut.txt' #评论处理后保存路径
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

l1 = len(data1)
l2 = len(data2)
l3 = len(data3)
l4 = len(data4)
l5 = len(data5)
l6 = len(data6)
l7 = len(data7)
l8 = len(data8)
l9 = len(data9)
l10 = len(data10)
l11 = len(data11)
l12 = len(data12)
data1 = pd.DataFrame(data1[0].unique())#去重
data2 = pd.DataFrame(data2[0].unique())
data3 = pd.DataFrame(data3[0].unique())
data4 = pd.DataFrame(data4[0].unique())
data5 = pd.DataFrame(data5[0].unique())
data6 = pd.DataFrame(data6[0].unique())
data7 = pd.DataFrame(data7[0].unique())
data8 = pd.DataFrame(data8[0].unique())
data9 = pd.DataFrame(data9[0].unique())
data10 = pd.DataFrame(data10[0].unique())
data11 = pd.DataFrame(data11[0].unique())
data12 = pd.DataFrame(data12[0].unique())

l13 = len(data1)
l14 = len(data2)
l15 = len(data3)
l16 = len(data4)
l17 = len(data5)
l18 = len(data6)
l19 = len(data7)
l20 = len(data8)
l21 = len(data9)
l22 = len(data10)
l23 = len(data11)
l24 = len(data12)

data1.to_csv(outputfile1, index = False, header = False)#文件输出
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

