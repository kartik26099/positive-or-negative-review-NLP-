import pandas as pd
import numpy as np
df=pd.read_csv(r"D:\coding journey\aiml\python\task\data set of ML project\natural language processing\test.txt",sep=";",header=None,names=["feeeling","emotion"])
import re
import nltk
nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
list=[]
print(df)
from collections import Counter # will count all the element in a coloumn
val_counts=Counter(df["emotion"]) 
print(val_counts)
indexes_to_drop = df[df['emotion'] == 'love'].index
df_dropped=df.drop(indexes_to_drop)
for i in range(0,df.shape[0]):
    feelings=re.sub("[^a-zA-z]"," ",df["feeeling"][i])
    feelings=feelings.lower()
    feelings=feelings.split()
    ps=PorterStemmer()
    feelings=[ps.stem(j)for j in feelings if j not in set(stopwords.words("english"))]
    feelings=" ".join(feelings)
    list.append(feelings)
print(list)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=3500)
x=cv.fit_transform(list).toarray()
y=df.iloc[:, -1]
from sklearn.model_selection import  train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.naive_bayes import GaussianNB
gb=GaussianNB()
gb.fit(x_train,y_train)
y_predict=gb.predict(x_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,y_predict))
from sklearn.svm import SVC
svc=SVC(kernel="linear")
svc.fit(x_train,y_train)
print(accuracy_score(y_test,svc.predict(x_test)))
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(x_train,y_train)
print(accuracy_score(y_test,lr.predict(x_test)))
