import pandas as pd
import numpy as np
df=pd.read_csv(r"D:\coding journey\aiml\python\udemy\Machine Learning A-Z (Codes and Datasets)\Part 7 - Natural Language Processing\Section 36 - Natural Language Processing\Python\Restaurant_Reviews.tsv",delimiter='\t',quoting=3)#i have used delimeter to specify it is a tsv filr and separted by a tab
#quoting is used to represent it to ignore all the "" as 3 represent ""
print(df)
# cleaning of data set
import re #help in cleaning the data set
import nltk #used to remove stop words(words which are not relevant for the algorithm)
nltk.download("stopwords") #download all the stop words


from nltk.corpus import stopwords # importing all the stop words
from nltk.stem.porter import PorterStemmer #this will help in steaming(simplifing a word for
# example loved get convert in love and if not done we will have two coloumn of love and loved)

corpus=[] # creating an empty list to store only cleaned element
#what we are doing in cleaning the data set

for i in range(0,df.shape[0]):
    review=re.sub("[^a-zA-Z]"," ",df["Review"][i]) #we are going to remove evry single punctuation by a space
       #"[^a-zA-Z]" this ^ means does not include any  so we are telling it to remove everything except letter a-z and A-z
    review=review.lower() #this will help in lowering all the leterrs
    review=review.split() #will split the sentence for steaming
    ps=PorterStemmer()
    all_words=stopwords.words("english")
    all_words.remove("not")
    review = [ps.stem(word) for word in review if word not in set(all_words)]
        # helped in steaming the words in column and we have not included stop words
    review=" ".join(review) # joing all the words which has been steammed

    corpus.append(review)
print(corpus)
from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer help in tokenizing
cv=CountVectorizer(max_features=1500)# it has an input maximum_feature which will take input as max wordsso
                    #before giving it make a dry run and find how many words are there
x=cv.fit_transform(corpus).toarray() #will toxanize all the words
y=df.iloc[:, -1].values
print(x.shape[1]) # before putting value :-we have total 1566 columns that means we have 1566 words in which we will consider
                #66 are not useful and put it as a input in CounterVectorizer which will tae words which has repeat only once or less
                  #after giving max_feature in cv it give 1500 words
#traing model

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
from sklearn.naive_bayes import GaussianNB
Gb=GaussianNB()
Gb.fit(x_train,y_train)
y_predict=Gb.predict(x_test)
from sklearn.metrics import confusion_matrix,accuracy_score
print(confusion_matrix(y_test,y_predict))
print(accuracy_score(y_test,y_predict))