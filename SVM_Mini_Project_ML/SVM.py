import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

svm = SVC()
fruits = pd.read_table('fruit_data_with_colors.txt')
fruits=fruits[fruits['fruit_subtype']!='unknown']
y=fruits['fruit_label']

print(fruits)
X=fruits[['mass','width','height','color_score']]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)
svm.fit(X_train, y_train)
y_pred=svm.predict(X_test)

print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(X_train, y_train)))
print('Accuracy of SVM classifier on test set: {:.2f}'
     .format(svm.score(X_test, y_test)))

"""
python3 SVM.py 
SVM.py:7: FutureWarning: read_table is deprecated, use read_csv instead, passing sep='\t'.
  fruits = pd.read_table('fruit_data_with_colors.txt')
    fruit_label fruit_name     fruit_subtype  mass  width  height  color_score
0             1      apple      granny_smith   192    8.4     7.3         0.55
1             1      apple      granny_smith   180    8.0     6.8         0.59
2             1      apple      granny_smith   176    7.4     7.2         0.60
3             2   mandarin          mandarin    86    6.2     4.7         0.80
4             2   mandarin          mandarin    84    6.0     4.6         0.79
5             2   mandarin          mandarin    80    5.8     4.3         0.77
6             2   mandarin          mandarin    80    5.9     4.3         0.81
7             2   mandarin          mandarin    76    5.8     4.0         0.81
8             1      apple          braeburn   178    7.1     7.8         0.92
9             1      apple          braeburn   172    7.4     7.0         0.89
10            1      apple          braeburn   166    6.9     7.3         0.93
11            1      apple          braeburn   172    7.1     7.6         0.92
12            1      apple          braeburn   154    7.0     7.1         0.88
13            1      apple  golden_delicious   164    7.3     7.7         0.70
14            1      apple  golden_delicious   152    7.6     7.3         0.69
15            1      apple  golden_delicious   156    7.7     7.1         0.69
16            1      apple  golden_delicious   156    7.6     7.5         0.67
17            1      apple  golden_delicious   168    7.5     7.6         0.73
18            1      apple       cripps_pink   162    7.5     7.1         0.83
19            1      apple       cripps_pink   162    7.4     7.2         0.85
20            1      apple       cripps_pink   160    7.5     7.5         0.86
21            1      apple       cripps_pink   156    7.4     7.4         0.84
22            1      apple       cripps_pink   140    7.3     7.1         0.87
23            1      apple       cripps_pink   170    7.6     7.9         0.88
24            3     orange     spanish_jumbo   342    9.0     9.4         0.75
25            3     orange     spanish_jumbo   356    9.2     9.2         0.75
26            3     orange     spanish_jumbo   362    9.6     9.2         0.74
27            3     orange  selected_seconds   204    7.5     9.2         0.77
28            3     orange  selected_seconds   140    6.7     7.1         0.72
29            3     orange  selected_seconds   160    7.0     7.4         0.81
30            3     orange  selected_seconds   158    7.1     7.5         0.79
31            3     orange  selected_seconds   210    7.8     8.0         0.82
32            3     orange  selected_seconds   164    7.2     7.0         0.80
33            3     orange      turkey_navel   190    7.5     8.1         0.74
34            3     orange      turkey_navel   142    7.6     7.8         0.75
35            3     orange      turkey_navel   150    7.1     7.9         0.75
36            3     orange      turkey_navel   160    7.1     7.6         0.76
37            3     orange      turkey_navel   154    7.3     7.3         0.79
38            3     orange      turkey_navel   158    7.2     7.8         0.77
39            3     orange      turkey_navel   144    6.8     7.4         0.75
40            3     orange      turkey_navel   154    7.1     7.5         0.78
41            3     orange      turkey_navel   180    7.6     8.2         0.79
42            3     orange      turkey_navel   154    7.2     7.2         0.82
43            4      lemon    spanish_belsan   194    7.2    10.3         0.70
44            4      lemon    spanish_belsan   200    7.3    10.5         0.72
45            4      lemon    spanish_belsan   186    7.2     9.2         0.72
46            4      lemon    spanish_belsan   216    7.3    10.2         0.71
47            4      lemon    spanish_belsan   196    7.3     9.7         0.72
48            4      lemon    spanish_belsan   174    7.3    10.1         0.72
/home/sarvesh/.local/lib/python3.5/site-packages/sklearn/svm/base.py:196: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
  "avoid this warning.", FutureWarning)
Accuracy of SVM classifier on training set: 0.90
Accuracy of SVM classifier on test set: 0.60
"""