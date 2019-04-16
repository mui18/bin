import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn.neighbors import KNeighborsClassifier 

names=['X','Y','Class']
dataset=pd.read_csv("data.csv",names=names)
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,2]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4)

testdata=np.array([[6,6]])
classifier=KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train,y_train)
print("Output of KNN\n")
y_pred=classifier.predict(testdata)
print(y_pred)
classifier=KNeighborsClassifier(n_neighbors=3,weights="distance")
classifier.fit(X_train,y_train)
print("Output of Weighted KNN\n")
y_pred=classifier.predict(testdata)
print(y_pred)


"""
Output of KNN
['Negative']
Output of Weighted KNN 
['Negative']

"""