#-*- coding: utf-8 -*-
import pandas as pd

#参数初始化
file = '../data/internet_finance_cut.txt' #输入的是一个词 一个词空格隔开的句子（comment）
stoplist = '../data/stoplist.txt'


data = pd.read_csv(file, encoding = 'utf-8', header = None) #读入数据
stop = pd.read_csv(stoplist, encoding = 'utf-8', header = None, sep = 'tipdm',engine='python')
#sep设置分割词，由于csv默认以半角逗号为分割词，而该词恰好在停用词表中，因此会导致读取出错
#所以解决办法是手动设置一个不存在的分割词，如tipdm。
stop = [' ', ''] + list(stop[0]) #Pandas自动过滤了空格符，这里手动添加

data[1] = data[0].apply(lambda s: s.split(' ')) #定义一个分割函数，然后用apply广播
data[2] = data[1].apply(lambda x: [i for i in x if i not in stop]) #逐词判断是否停用词，思路同上

#print(data[2])
from gensim import corpora, models


#主题分析
neg_dict = corpora.Dictionary(data[2]) #建立词典
print(data[2])#一个句子是一个list
print(type(data[2])) #<class 'pandas.core.series.Series'> 其实就是一个下标 加上一个句子数组
#所以如果要输入新的句子 只需要把句子的格式转化为series的格式就好了



neg_corpus = [neg_dict.doc2bow(i) for i in data[2]] #建立语料库，用于自动推出文档结构，以及它们的主题等，也可称作训练语料

neg_lda = models.LdaModel(neg_corpus, num_topics = 20, id2word = neg_dict) #LDA模型训练
#这证明了LDA model自动将word转化成了word vector
print("this is topics:")

for i in range(20):
  print(neg_lda.print_topic(i))
for i in range(20):
  topic=neg_lda.print_topic(i).replace("0", "")
  topic=topic.replace("1", "")
  topic=topic.replace("2", "")
  topic=topic.replace("3", "")
  topic=topic.replace("4", "")
  topic=topic.replace("5", "")
  topic=topic.replace("6", "")
  topic=topic.replace("7", "")
  topic=topic.replace("8", "")
  topic=topic.replace("9", "")
  topic=topic.replace("*", "")
  topic=topic.replace(".", "")
  topic=topic.replace("+", "")
  topic=topic.replace("\"", "")
  topic=topic.replace("“", "")
  topic=topic.replace("”", "")
  #topic=topic.replace("行业", "")
  #topic=topic.replace("美元", "")
  #topic=topic.replace("亿", "")
  #topic=topic.replace("元", "")
  topic=topic.replace("网贷", "")
  #topic=topic.replace("金融", "")
  #topic=topic.replace("经济", "")
  
  
  print("the topic "+str(i)+" is:")
  print(topic)

#print(neg_lda.show_topics(num_topics=20)) #每个topic从0开始计数
  #print(type(neg_lda.print_topic(i)))#输出每个主题

#对新的文档的主题识别：
#print(neg_corpus[3])
#print(list(neg_dict))
  
#一个新的文章
new_li=[["持续","加强","私募","监管","力度","坚决","整治","行业","乱象","银监会"]]
from pandas import Series
obj = Series(new_li)
new_corpus = [neg_dict.doc2bow(i) for i in obj]


print(str(list(neg_lda.get_document_topics(bow=new_corpus))))



#为什么每次运行LDA模型结果都不一样？相同的源数据
#解答：由于LDA使用随机性的训练和推理步骤 通过重置numpy.random种子可以产生稳定的主题
