from sklearn.cluster import KMeans,MiniBatchKMeans
from sklearn.externals import joblib
import numpy
import time
import pandas as pd  
import matplotlib.pyplot as plt  
#读取文本数据到DataFrame中，将数据转换为matrix，保存在dataSet中  
df = pd.read_excel(r'C:\Users\Dominic\Desktop\DATA\Clustering data.xlsx')
#print(df)
df =df.drop(['Country','Year','Rank','Total'],axis=1)
print(df)
X = df.as_matrix(columns=None)  


from matplotlib import style

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3,max_iter=10000,algorithm='elkan')

kmeans.fit(X)

centroid = kmeans.cluster_centers_
labels = kmeans.labels_

#print (centroid)


colors = ["r.","g.","c."]
for i in range(len(X)):
   #print ("coordinate:" , X[i], "label:", labels[i])
   if i%1==0:
      plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=5,marker = '.',label="Fragile States")
   if i%2==1:
      plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=5,marker = '.',label="Vulnerable States")
   if i%3==2:
      plt.plot(X[i][0],X[i][1],colors[labels[i]],markersize=5,marker = '.',label="Stable States")
      
plt.legend(loc='lower right')
plt.scatter(centroid[:,0],centroid[:,1], marker = "x", s=150, linewidths = 5, zorder =10)

plt.show()
