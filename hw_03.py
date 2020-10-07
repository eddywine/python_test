# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 12:22:26 2019

@author: emukhebi
"""

import pandas as pd
import nltk
import numpy as np
import re  
import nltk  
import matplotlib.pyplot as plt  
import os
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from string import punctuation, digits
from IPython.core.display import display, HTML
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import sent_tokenize
import tarfile
import sklearn.utils
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn import model_selection
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report



extract_tarfile = tarfile.open("hw_03_train_data.tar.xz")
extract_tarfile.extractall()
positive_train=open("pos_train.txt", 'r',encoding='utf-8')
positive_train=positive_train.readlines()
#print(positive_train)
negative_train=open("neg_train.txt", 'r',encoding='utf-8')
negative_train=negative_train.readlines()


# Removing stop words int
from nltk.corpus import stopwords
no_stop_words=[]
english_stop_words = stopwords.words('english')
def remove_stop_words(corpus):
    removed_stop_words = []
    for review in corpus:
        removed_stop_words.append(
            ' '.join([word for word in review.split() 
                      if word not in english_stop_words])
        )
    return removed_stop_words

positive_train = remove_stop_words(positive_train)
negative_train = remove_stop_words(negative_train)


# Normalization i.e through the use of stemming and lematization
# stemming
def get_stemmed_text(corpus):
    from nltk.stem.porter import PorterStemmer
    stemmer = PorterStemmer()
    return [' '.join([stemmer.stem(word) for word in review.split()]) for review in corpus]

positive_train = get_stemmed_text(positive_train)
negative_train = get_stemmed_text(negative_train)

# Lematization of the input data

def get_lemmatized_text(corpus):
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    return [' '.join([lemmatizer.lemmatize(word) for word in review.split()]) for review in corpus]

positive_train = get_lemmatized_text(positive_train)
negative_train = get_lemmatized_text(negative_train)
#print(no_stop_words)
#print(lemmatized_reviews)




# Opening the input file that will be used in the prediction
# opening the file for the input data that will be used for prediction
input_text=open("input.txt", 'r',encoding='utf-8')
input_text=input_text.readlines()
input_text = [(line,'test') for line in input_text]
testing_data=pd.DataFrame.from_records(input_text)
testing_data.columns=["sentiment","labels"]

print(testing_data.shape[0])



# putting labels in the dataset
positive_train = [(line,  1) for line in positive_train]
negative_train = [(line,  -1) for line in negative_train]
#all_data=[positive_train,negative_train]
df_negative=pd.DataFrame.from_records(negative_train)
df_positive=pd.DataFrame.from_records(positive_train)
final=pd.concat([df_negative, df_positive])
final.columns=["sentiment","labels"]
# Shuffle Pandas data frame

final_shuffled_data = sklearn.utils.shuffle(final)
print("the file is printing")


# splitting the data and creating the model

print("the file is printing2")
# creating a feature vectors
labels=final_shuffled_data['labels']
train_data,test_data,y_train,y_test= train_test_split(final_shuffled_data,labels,test_size=0.2, random_state=0)
vectorizer=TfidfVectorizer(smooth_idf=True,norm='l2',use_idf=True,sublinear_tf=True)

train_data=vectorizer.fit_transform(train_data['sentiment'])
test_data = vectorizer.transform(test_data['sentiment'])
testing_data=vectorizer.transform(testing_data['sentiment'])# used as the actual blind test file for the model


# logistic linear regression

model = LogisticRegressionCV(random_state=0, solver='lbfgs')
model.fit(train_data,y_train)# fit data int the model
model.score(train_data,y_train)

predictions = model.predict(test_data)

print(accuracy_score(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
predictions_inputdata = model.predict(testing_data)

"""
# creating the output file 
import bz2
f=bz2.open("output.txt.bz2","wb")
#f.write(predictions_inputdata)
np.savetxt(f,predictions_inputdata,fmt='%1.0f\n')
f.close()
"""

f=open("output.txt","w")
#f.write(predictions_inputdata)
np.savetxt(f,predictions_inputdata,fmt='%1.0f')
f.close()

import os
import bz2
myCmd ='bz2 -z output.txt'
os.system(myCmd)





