
# coding: utf-8

# # Information Retrieval Final Project
# ## TweetQuake
# Detection of Earthquake using Twitter data.
# Twitter users are everywhere, check the twitter map and earthquake frequency map
# <img src="img/TwitterMap.png" alt="Drawing" style="width: 475px;float:left; margin-top: 30px" title = "Twitter Map"/>
# <img src="img/EarthQuakeMap.png" alt="Drawing" style="width: 475px;float:right" title = "Earthquake Map"/>
# <img src="img/EarthQuakeTweet.png" alt="Drawing" style="width: 600px;" title = "Earthquake Tweet Frequency"/>

# In[8]:

import re
import csv
import nltk
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from nltk.stem.porter import PorterStemmer
get_ipython().magic(u'matplotlib inline')

def get_data(filename):
    df = pd.read_csv(filename)
    return df
def length(df):
    return len(df['Tweet_Text'])


data = get_data('2013_Bohol_earthquake-tweets_labeled.csv')
data['Info'] = 'related'
data.Info[('Not related' == data.Informativeness)] = 'not-related'
data['Tweet_Text'] = data['Tweet_Text'].apply(lambda x: x.decode('unicode_escape').                                          encode('ascii', 'ignore').                                          strip())

# data.head()
X = data[['Tweet_ID','Tweet_Text','Info']]
y = data.Info
porter_stemmer = PorterStemmer()

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
wordmap = defaultdict(int)

def Stopword(tweet):
    nostop = str()
    for word in tweet:
        #word = word.decode('utf-8')
        if word not in stopwords: nostop+= " " + word
    return nostop

def remove_stopword (X):
    X['nostopword'] = X.Tweet_Text.str.split().apply(Stopword)
    return X

def Porter_Stem(tweet):
    stemmed_word = str()
    for word in tweet:
        word = word.decode('utf-8')
        stemmed_word += " " + (porter_stemmer.stem(word))
    return stemmed_word
            
def stemming (X):
    X['stem'] = X.nostopword.str.split().apply(Porter_Stem)
    return X

def update_wordmap(tweet):
    #update wordmap
    for word in tweet:
        wordmap[word]+=1

def term_frequency_plot(X):
    #Plot a graph for term frequency in tweets
    X.stem.str.split().apply(update_wordmap)
    sorted_x = sorted(wordmap.items(), key=operator.itemgetter(1), reverse = True)
    objects = list()
    freq = list()
    for i in range(10):
        objects.append(sorted_x[i][0])
        freq.append(sorted_x[i][1])

    x_pos = np.arange(len(objects))
    plt.barh(x_pos, freq, align='center', alpha=0.5)
    plt.xlabel("Frequency")
    plt.yticks(x_pos, objects)
    plt.title('Term Frequency usage')
    plt.show()
    
# Feature Extraction
# X['count'] = X.Tweet_Text.str.split(' ').apply(len)#.value_counts()
def has_earthquake(val):
    for i in range(len(val)):
        if val[i].lower().find('earthquake') != -1 or val[i].lower().find('quake') != -1 or val[i].lower().find('shaking') != -1:
            return 1
    return 0
X['has_earthquake'] = X.Tweet_Text.str.split().apply(has_earthquake)
def Feature_extraction_A(X):
    X['total_words'] = X.Tweet_Text.str.split(' ').apply(len)
    X['position_query_word'] = X.Tweet_Text.str.split().apply(find_position)
#     X['feature_a'] = X.temp_1.astype(str) + ' words, the ' + X.temp_2.astype(str) + ' word'
#     X = X.drop('temp_1', axis=1)
#     X = X.drop('temp_2', axis=1)
    return X

def Feature_extraction_B(X):
    X['feature_b'] = X.Tweet_Text.str.split().apply(remove_punc)
    return X

def Feature_extraction_C(X):
    X['feature_c'] = X.Tweet_Text.str.split().apply(find_before_after_query_word)
    return X

def remove_punc(value):
    punctuation_marks = re.compile(r'[.?!,":;#-]')
    words = []
    for each in value:
        words.append(punctuation_marks.sub('',each))
    return ','.join(words)

def find_position(val):
    for i in range(len(val)):
        if val[i].lower().find('earthquake') != -1:
            return i
    return -1

def find_before_after_query_word(val):
    for i in range(len(val)):
        if val[i].lower().find('earthquake') != -1:
            if i == 0 and len(val)>1:
                return ','+val[i+1]
            elif i == len(val)-1 and len(val)>1:
                return val[i-1]+','
            else:
                return val[i-1]+','+val[i+1]
    return ', '
    
# X['position'] = X.Tweet_Text.str.split().apply(find_position)
# X['before_query1'] = 0
# X.before_query1[X.position > 0] = X.position - 1
X = remove_stopword(X)
X = stemming (X)
term_frequency_plot(X)
X = Feature_extraction_A(X)
# X = Feature_extraction_B(X)
# X = Feature_extraction_C(X)
# X

y = data['Info'].values
X = X.drop('Tweet_ID',axis=1)
X = X.drop('Tweet_Text',axis=1)

X = X.drop('nostopword',axis=1)
X = X.drop('stem',axis=1)


X = X.drop('Info',axis=1)
X
# X = X.drop('total_words',axis=1)
# X = X.drop('position_query_word',axis=1)



# ## Integration of Twitter API
# Keys are for reference 
# We can get new data set using this
# or modify the old data set to get some new information

# In[10]:

import twitter, json
api = twitter.Api(consumer_key='4j8Uk7Hea3pdgEuJ6nkvqvVYO',
                  consumer_secret='VU4nTJFz4KQSr66Q1MH7snV0BFEkkeL8sOnu5qGSfzU9poUSU1',
                  access_token_key='44338623-psdoVV5cnUnS9TN0fgrtt4KoMfwxXGfevzS5CllRu',
                  access_token_secret='VKioM8alAKMPTlE1BauuzLC1SLtXbpDWZZ6qEDPi8xz3F')
results = api.GetSearch(
    raw_query="q=-earthquake%2C%20-shaking%2C%20-quake%2C%20-tremor%20since%3A2010-09-15%20until%3A2017-04-23&src=typd&lang=en")#"q=earthquake%20&result_type=recent&since=2016-07-19&count=10&lang=en")

# f = csv.writer(open("test_neg1.csv", "wb+"))
# f.writerow(["tweet_id", "tweet_text", "created_at", "url", "location","media"])
current_id = ''
# print results
for x in results:
#     print x.text
    x.text = x.text.encode('utf-8').strip()
    x.user.location = x.user.location.encode('utf-8').strip()
#     if x.media is not None:
#     x.media = x.media.encode('utf-8').strip()
#     if x.urls is not None and x.media is not None:
#         f.writerow([x.id,x.text,x.created_at,1,x.user.location,1])
#     elif x.urls is None and x.media is not None:
#         f.writerow([x.id,x.text,x.created_at,0,x.user.location,1])
#     elif x.urls is not None and x.media is None:
#         f.writerow([x.id,x.text,x.created_at,1,x.user.location,0])
#     else:
#         f.writerow([x.id,x.text,x.created_at,0,x.user.location,0])
#     f.writerow([x.id,x.text,x.created_at,x.urls,x.user.location,x.media])
    current_id = x.id
count = 0
for i in range(0):
    results = api.GetSearch(
        raw_query="q=-earthquake%2C%20-shaking%2C%20-quake%2C%20-tremor%20since%3A2010-11-18%20until%3A2017-04-23&src=typd&count=100&lang=en&since_id"+str(current_id))#"q=earthquake%20&result_type=recent&since=2016-07-19&count=10&lang=en")
    for x in results:
        x.text = x.text.encode('utf-8').strip()
        x.user.location = x.user.location.encode('utf-8').strip()
#         x.media = x.media.encode('utf-8').strip()
#         if x.urls is not None and x.media is not None:
#             f.writerow([x.id,x.text,x.created_at,1,x.user.location,1])
#         elif x.urls is None and x.media is not None:
#             f.writerow([x.id,x.text,x.created_at,0,x.user.location,1])
#         elif x.urls is not None and x.media is None:
#             f.writerow([x.id,x.text,x.created_at,1,x.user.location,0])
#         else:
#             f.writerow([x.id,x.text,x.created_at,0,x.user.location,0])
        f.writerow([x.id,x.text,x.created_at,x.urls,x.user.location,x.media])
        current_id = x.id
        count += 1
#     print count

# Find tweets using tweet ID
res = api.GetStatus(389949367009808384)

# Code to get the new data with Location and Created date
# def find_created_at(id):
#     try:
#         res = api.GetStatus(id)
#         return res.created_at
#     except:
#         return ''
# X['created_at'] = X.Tweet_ID.astype(int).apply(find_created_at)
# def find_location(id):
#     try:
#         return api.GetStatus(id).user.location
#     except:
#         return ''
# X['location'] = X.Tweet_ID.astype(int).apply(find_location)
# X.to_csv('new_data.csv', sep=',',encoding='utf-8')


# ## SVM Model Implementation

# In[9]:

## SVM model implementation
import math
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction import DictVectorizer


# vec = DictVectorizer()
# df = vec.fit_transform(X.feature_a.str).toarray()
# X = pd.get_dummies(X.feature_b.str.split())

def standardizedX(X):
    scaler = StandardScaler().fit(X)
    standardizedX = scaler.transform(X)
    return standardizedX
# Tune hyperparameter gamma and choose best gamma for model training
def hyperparameter_tuning(X, y):
	# Choose value of hyper parameter from below values of gamma
    gammas = [2**-1, 2**-3, 2**-5, 2**-7, 2**-9]
    classifier = GridSearchCV(estimator=svm.SVR(), cv=10, param_grid=dict(gamma=gammas))

    kf = KFold(n_splits=10, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        classifier.fit(X_train, y_train)
    return classifier

# 10- fold cross validation and error evaluation
# Loop of 30 to see 300 validations with shuffle on and off
# Loop of 10 to check the confidence interval - paper doesn't describe much on the 
# confidence interval. This works only for shuffle On.
def cross_validation_evaluation(X, y):
    mean_error, mad_error = 0, 0
    count = 0
    mean_min, mad_min = 100, 100
    mean_max, mad_max = 0, 0
    f1_sco,accuracy_sco = 0,0
#     classifier = hyperparameter_tuning(X, y)
    model=svm.SVC(kernel='linear', gamma=0.001)#classifier.best_estimator_.gamma)
    
    for j in range(1):
        for i in range(10):
            kf = KFold(n_splits=10, random_state=None, shuffle=True)
#             print len(X)
            for train_index, test_index in kf.split(X):
                count += 1
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(X_train,y_train)
#                 mean_error += mean_squared_error(y_test, model.predict(X_test))
#                 mad_error += mean_absolute_error(y_test,model.predict(X_test))
#                 print (y_test,model.predict(X_test))
#                 print (model.predict(X_test))
                f1_sco += f1_score(y_test,model.predict(X_test),pos_label='related')
                accuracy_sco += accuracy_score(y_test,model.predict(X_test),normalize=True)
#                 print classification_report(y_test,model.predict(X_test))
#         mean_min = min(mean_min, (mean_error/count)**0.5)
#         mean_max = max(mean_max, (mean_error/count)**0.5)
#         mad_min = min(mad_min, (mad_error/count))
#         mad_max = max(mad_max, (mad_error/count))
#     RMSE = (mean_error/count)**0.5
#     MAD = mad_error/count
    print 'F1-score: ' + str(f1_sco/count)
    print 'Accuracy: ' + str(accuracy_sco/count)
#     return RMSE, MAD, mean_min, mean_max, mad_min, mad_max
    

# X = standardizedX(X)
# print X.tail()
cross_validation_evaluation(X.values, y)
# RMSE, MAD, mean_min, mean_max, mad_min, mad_max = cross_validation_evaluation(X.values, y)
# print "For feature: "
# print 'RMSE +- Confidence Interval: '+"{0:.2f}".format(RMSE)+' +- '+"{0:.2f}".format(max(RMSE-mean_min, mean_max-RMSE))
# print 'MAD +- Confidence Interval: '+"{0:.2f}".format(MAD)+' +- '+"{0:.2f}".format(max(MAD-mad_min, mad_max-MAD))


# In[ ]:



