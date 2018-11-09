import gensim, logging#载入gensim包#
import os
import csv
csvFile=open(r"C:\Users\Arthur\Desktop\vivomidcut.csv","w+",newline="",encoding='utf-8')#在桌面形成csv文件#
writer=csv.writer(csvFile)

logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)#记录日志#
sentences = gensim.models.doc2vec.TaggedLineDocument('vivomidcut.txt')#输入文件vivomidcut.txt的格式就是每个doc 对应内容的分词，空格隔开，每个doc是一行#
#用TaggedLineDocument 实现，每个doc默认编号#
model = gensim.models.Doc2Vec(sentences, size = 50, window = 5)#size表示生成的向量的维度，一般为100维#
#window表示训练的窗口的大小也就是训练数据周围读取了几个数据#
model.save('review_pure_text_model.txt')#以txt格式存储#
print(len(model.docvecs))#打印文本长度#
out = open('review_pure_text_vector.txt', 'w')
for idx, docvec in enumerate(model.docvecs):#用enumerate函数遍历文本#
    for value in docvec:
      out.write(str(value) + ' ')
    out.write('\n')
    writer.writerow(docvec.tolist())#将输出改为数组形式，并输出#
out.close
csvFile.close()#关闭csv#
