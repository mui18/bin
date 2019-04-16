import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn import metrics 
from sklearn.externals.six import StringIO 
from sklearn.tree import export_graphviz
from IPython.display import Image
import pydotplus

col_names=['Age','Income','Gender','Martial_status','Buys']
data=pd.read_csv("data.csv",header=None,names=col_names)

cleanup_names={"Age":{"<21":0, "21-35":1, ">35":2},
"Income":{"Low":0, "Medium":1 ,"High":2},
"Gender":{"Male":0, "Female":1},
"Martial_status":{"Single":0,"Married":1},
"Buys":{"Yes":0,"No":1}}

data.replace(cleanup_names,inplace=True)

feature_cols=['Age','Income','Gender','Martial_status']
X=data[feature_cols]
y=data.Buys
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,y,test_size=0.4)
clf=DecisionTreeClassifier()
clf=clf.fit(X_Train,Y_Train)
predict=clf.predict(X_Test)
print("Accuracy",metrics.accuracy_score(Y_Test,predict))
dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,special_characters=True,feature_names=feature_cols,class_names=['0','1'])
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png("bus.png")
Image(graph.create_png())