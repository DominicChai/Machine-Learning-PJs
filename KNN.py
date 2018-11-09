from sklearn.datasets import load_digits  
from sklearn import neighbors  
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
iris = load_digits()
#print(iris)
x=iris.data
y=iris.target
#print(x)
#print(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=1)
#print(y_test)
for i in [1,3,5]:
#查看iris数据集  
    knn = neighbors.KNeighborsClassifier(n_neighbors=i,p=2)  
    #训练数据集  
    knn.fit(x_train,y_train)  
    #预测  
    result=accuracy_score(y_test,knn.predict(x_test))
    print("accuracy_score for k-neighbor "+str(i)+" is: ")
    print(result)
