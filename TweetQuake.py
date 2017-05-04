
# coding: utf-8

# # Information Retrieval Final Project
# ## TweetQuake
# Detection of Earthquake using Twitter data.
# Twitter users are everywhere, check the twitter map and earthquake frequency map
# <img src="img/TwitterMap.png" alt="Drawing" style="width: 475px;float:left; margin-top: 30px" title = "Twitter Map"/>
# <img src="img/EarthQuakeMap.png" alt="Drawing" style="width: 475px;float:right" title = "Earthquake Map"/>
# <img src="img/EarthQuakeTweet.png" alt="Drawing" style="width: 600px;" title = "Earthquake Tweet Frequency"/>

# ## Introduction and Problem Statement
# 
# Online Social Media have become an important alternate channel to spread information during natural disasters and other important events. Twitter is one such channel where people look forward to knowing about their friends and relatives after a catastrophic event such as an Earthquake, tornado, and others. The real-time nature of Twitter enables it to be an important source for information extraction. In a case of an earthquake, people who experience it, tweet about it to inform their friends about their safety and to spread the news of a target event. In this report, we have investigated the extraction of event information about Earthquake using the service API of Twitter. Our algorithm classifies Tweets on the basis of whether they are relevant to Earthquake and if yes, whether they are indicative of a current earthquake.
# 
# 
# Earlier, the main source of information for an earthquake like event was the radio, however, there was some time lapse from the occurrence of an event and the announcement and also the reach of information was not global. In the past few years, people especially youngsters are turning to Twitter to learn about such events. Twitter has the advantage of fast propagation speed and more number of contributors when compared to radio. 
# 
# Twitter is categorized as a microblogging service, ie a form of blogging that enables users to send brief text and multimedia updates. Other similar services are Tumblr, Plurk etc. We focus on examining the data received from Twitter due to its popularity and data volume. There are 190 million people who use Twitter per month, generating 65 million tweets per day[1] . Compared to other microblogging sites, Twitter user updates their account more often and tweet regularly, often several times in a day. For example, when an airplane crash-landed on the Hudson River in New York, the first published report was via Twitter. Similarly, many events such as presidential campaigns, football games, and others are being analyzed more often via social media tools such as Twitter rather than radio or News channel. 
# 
# Generally, people who experience an event such as Earthquake are the first to report it via Twitter, and the tweets spread out even before the event is registered with the USGS and long before it is covered in news channels. The motivation behind our research was:
# 	Can we detect such event occurrence in real-time by monitoring tweets?
# The amount of real-time information flow via Twitter motivated us to design a real-time automated Earthquake Detection Classifier. 
# 
# ## Related Work.
# 
# Event or disaster detection using twitter is one of the fastest and effective way to get the relevant information. Many researchers used the tweets to get the information about the trends, social media relations of twitter users, retweet activities. Earthquake detection using tweets is first observed in 2010 (Tweet Analysis for Real-Time Event Detection and Earthquake Reporting System Development
# )[8]
# Since than USGS developed an application to effectively detect an earthquake using as minimum as 14 tweets. Currently, twitter is being used to detect an earthquake along with other disasters. \
# 
# Previously many researchers have published studies of twitter and observed the network structure of Twitter [2], [3], [4]. Some of the researchers used social media as of the characteristics to analyze the Twitter data [5], [6]. In papers [7] and [8], authors have development different applications using the tweet and other relevant data of Twitter and developed some real time applications.
# 
# 
# Most of the papers on event detection use basic features like presence of a query word like earthquake or shaking along with the length of tweet (as the person experiencing tweet will not write a big and fancy tweet). The accuracy of the previous event detection as described in the paper [8] is just 76% in best case. We have improved the feature selection process and used different classifiers to detect the earthquake relevant tweets and current earthquake related tweets to improve the accuracy. 

# ## Approach / Methods.
# 
# We have used SVM classifier to classify the tweets. 
# 
# Data Collection
# We have collected previous earthquake related data from different below sources
# http://crisislex.org/data-collections.html
# Twitter dev API
# https://earthquake.usgs.gov/earthquakes/browse/significant.php 
# Manual labeling of tweets from Twitter.
# 
# ### 3 Unique Contributions (Compared to the research paper)
# 1. Two-step classification - To mimic the structure of classes in data.
# 2. Using NBSVM classifer  - A combination of Naive Bayes and SVM.
# 3. Creative features for data related to our domain.
# 
# Since it was difficult to get proper training data of good quality for this problem, we manuall collected the Twitter data by querying the earthquake time tweets which contains some information about earthquake and manually labelled than as earthquake indicative or not. 
# 
# Initial Exploration
# Initially the we used the data collected from crisislex and observed very high accuracy as the earthquake relevant data has earthquake keyword and the irrelevant tweet does not have the keyword. We observed that the Model was overfitting the data. After taking the effort to manually collect and label data, our classifier improved.
# 
# Revised/Different approaches
# The approach described in the paper was to find three basic features like the presence of query word, the position of query word in the tweet and the words in the tweet along with the word before and after the query word. The data used in the paper is all positive samples and the final accuracy was 73 percent which is same by using only the feature A (presence of query word in the tweet) and using all three features. 
# 
# We used two different classifiers to find the earthquake 'related' and earthquake 'relevant' tweet and the accuracy observed is 87 percent by using the best training data set. Our model runs in real time by getting the tweets using twitter API and classifying them as earthquake related.

# In[2]:

get_ipython().magic(u'matplotlib inline')
import re
import csv
import nltk
import operator
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
from collections import defaultdict
from IPython.core.interactiveshell import InteractiveShell
from sklearn.feature_extraction.text import TfidfVectorizer
InteractiveShell.ast_node_interactivity = "all"
from nltk.stem.porter import PorterStemmer
import warnings
warnings.filterwarnings('ignore')

def get_data(filename):
    df = pd.read_csv(filename)
    return df

def length(df):
    return len(df['Tweet_Text'])


data = get_data('2013_Bohol_earthquake-tweets_labeled.csv')
data['Info'] = 'related'
data.Info[('Not related' == data.Informativeness)] = 'not-related'
data['Tweet_Text'] = data['Tweet_Text'].apply(lambda x: x.decode('unicode_escape').                                          encode('ascii', 'ignore').                                          strip().lower())

X1 = data[['Tweet_ID','Tweet_Text','Info']]
y1 = data.Info
porter_stemmer = PorterStemmer()

# load nltk's English stopwords as variable called 'stopwords'
stopwords = nltk.corpus.stopwords.words('english')
wordmap = defaultdict(int)

def Stopword(tweet):
    nostop = []
    for word in tweet:
        #word = word.decode('utf-8')
        if word not in stopwords: nostop.append(word)
    return ' '.join(nostop)

def remove_stopword (X):
    X['nostopword'] = X.Tweet_Text.str.split().apply(Stopword)
    return X

def Porter_Stem(tweet):
    stemmed_word = []
    for word in tweet:
        word = word.decode('utf-8')
        stemmed_word.append(porter_stemmer.stem(word)) 
    return ' '.join(stemmed_word)
            
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
    
def Feature_extraction_A(X):
    X['total_words'] = X.stem.str.split(' ').apply(len)
    X['position_query_word'] = X.stem.str.split().apply(find_position)
    return X

def Feature_extraction_BnC(X):
    word_features = X[['Tweet_ID','Tweet_Text','Info']]
    word_features = word_features.values.tolist()

    data_pos = []
    data_neg = []

    for tweet in word_features:
        if tweet[2] == 'related':
            data_pos.append(tweet[1])
        else:
            data_neg.append(tweet[1])

    token_pattern = r'\w+|[%s]' % string.punctuation

    vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                                 token_pattern=token_pattern,
                                 binary=True,
                                max_features=5000)
    word_vector = vectorizer.fit_transform(data_pos+data_neg)
    
    return X.join(pd.DataFrame(word_vector.toarray()))

def find_position(val):
    for i in range(len(val)):
        if val[i].lower().find(porter_stemmer.stem('earthquake')) != -1:
            return i
    return -1


X1 = remove_stopword(X1)
X1 = stemming (X1)
term_frequency_plot(X1)
X1 = Feature_extraction_A(X1)
# X1 = Feature_extraction_BnC(X1)

y1 = data['Info'].values
X1 = X1.drop('Tweet_ID',axis=1)
X1 = X1.drop('Tweet_Text',axis=1)
X1 = X1.drop('nostopword',axis=1)
X1 = X1.drop('stem',axis=1)
X1 = X1.drop('Info',axis=1)
X1 = X1.values
# X


# In[ ]:

type(X1)
type(y1)


# ## SVM Model Implementation

# In[3]:

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import train_test_split
import sklearn.metrics


scaler1 = preprocessing.StandardScaler().fit(X1)
X1 = scaler1.transform(X1)

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.20, random_state=42)


# In[4]:

C_range = np.logspace(-2, 10, 5, base=2)
gamma_range = np.logspace(-9, 1, 5, base=2)
# k_options = ['linear','poly','rbf']
params_grids = dict(gamma=gamma_range, C=C_range)
grid1 = GridSearchCV(SVC('rbf'), param_grid=params_grids, cv=10)
grid1.fit(X1_train,y1_train)

print grid1.best_params_
# grid1.grid_scores_


# In[5]:

clf1 = SVC(kernel='rbf' ,C=grid1.best_params_['C'], 
             gamma=grid1.best_params_['gamma'])

clf1.fit(X1_train,y1_train)
y1_pred = clf1.predict(X1_test)

sklearn.metrics.accuracy_score(y1_test, y1_pred)
sklearn.metrics.precision_score(y1_test, y1_pred,pos_label='related')
sklearn.metrics.recall_score(y1_test, y1_pred,pos_label='related')
sklearn.metrics.f1_score(y1_test, y1_pred,pos_label='related')


# ## Naive-Bayes SVM Implementation

# The Naive-Bayes SVM (NBSVM) is a simple but novel SVM variant using NB log-count ratios as feature values and is supposed to be a robust performer. This model is an interpolation between MNB and SVM, which can be seen as a form of regularization: trust NB unless the SVM is very confident. 
# 
# The concept of NBSVM has been obtained from this research paper : Wang, Sida, and Christopher D. Manning. "Baselines and bigrams: Simple, good sentiment and topic classification." Proceedings of the 50th Annual Meeting of the Association for Computational Linguistics: Short Papers-Volume 2. Association for Computational Linguistics, 2012.
# 
# The class implementation of the NBSVM has been obtained from this repository: https://github.com/Joshua-Chin/nbsvm.git
# 
# For the first classification, where we are classifying earthquake-relevant tweets from earthquake-irrelevant tweets, we are using the NBSVM classifier as it's giving us a higher accuracy compared to the SVM classifier discussed above. 

# In[6]:

from scipy.sparse import spmatrix, coo_matrix
from sklearn.base import BaseEstimator
from sklearn.linear_model.base import LinearClassifierMixin, SparseCoefMixin
from sklearn.svm import LinearSVC

class NBSVM(BaseEstimator, LinearClassifierMixin, SparseCoefMixin):

    def __init__(self, alpha=1, C=1, beta=0.25, fit_intercept=False):
        self.alpha = alpha
        self.C = C
        self.beta = beta
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if len(self.classes_) == 2:
            coef_, intercept_ = self._fit_binary(X, y)
            self.coef_ = coef_
            self.intercept_ = intercept_
        else:
            coef_, intercept_ = zip(*[
                self._fit_binary(X, y == class_)
                for class_ in self.classes_
            ])
            self.coef_ = np.concatenate(coef_)
            self.intercept_ = np.array(intercept_).flatten()
        return self

    def _fit_binary(self, X, y):
        p = np.asarray(self.alpha + X[y == 1].sum(axis=0)).flatten()
        q = np.asarray(self.alpha + X[y == 0].sum(axis=0)).flatten()
        p = np.asarray(p,dtype=np.float)
        q = np.asarray(q,dtype=np.float)
        r = np.log(p/np.abs(p).sum()) - np.log(q/np.abs(q).sum())
        b = np.log((y == 1).sum()) - np.log((y == 0).sum())

        if isinstance(X, spmatrix):
            indices = np.arange(len(r))
            r_sparse = coo_matrix(
                (r, (indices, indices)),
                shape=(len(r), len(r))
            )
            X_scaled = X * r_sparse
        else:
            X_scaled = X * r

        lsvc = LinearSVC(
            C=self.C,
            fit_intercept=self.fit_intercept,
            max_iter=10000
        ).fit(X_scaled, y)

        mean_mag =  np.abs(lsvc.coef_).mean()

        coef_ = (1 - self.beta) * mean_mag * r +                 self.beta * (r * lsvc.coef_)

        intercept_ = (1 - self.beta) * mean_mag * b +                      self.beta * lsvc.intercept_

        return coef_, intercept_


# Here we are diving the data into a train-test split of 80-20. We are building our vocabulary based on the training data using scikit-learn's method TfidfVectorizer. We are using unigram, bigram and trigram features and the Tf-Idf weighting scheme on the word vector. We are setting the tf term in tf-idf to be binary, as it was increasing our accuracy.

# In[7]:

import string
from sklearn.feature_extraction.text import TfidfVectorizer

print("Vectorizing Training Text")

X2 = data[['Tweet_ID','Tweet_Text','Info']]
X2 = X2.values.tolist()

data_pos = []
data_neg = []

for tweet in X2:
    if tweet[2] == 'related':
        data_pos.append(tweet[1])
    else:
        data_neg.append(tweet[1])

train_pos = data_pos[:1100]
train_neg = data_neg[:1100]

token_pattern = r'\w+|[%s]' % string.punctuation

vectorizer = TfidfVectorizer(ngram_range=(1, 3),
                             token_pattern=token_pattern,
                             binary=True)

X2_train = vectorizer.fit_transform(train_pos+train_neg)
y2_train = np.array([1]*len(train_pos)+[0]*len(train_neg))

print("Vocabulary Size: %s" % len(vectorizer.vocabulary_))
print("Vectorizing Testing Text")

test_pos = data_pos[1100:]
test_neg = data_neg[1100:]

X2_test = vectorizer.transform(test_pos+test_neg)
y2_test = np.array([1]*len(test_pos)+[0]*len(test_neg))

print("Fitting Model")

mnbsvm = NBSVM()
mnbsvm.fit(X2_train, y2_train)
print('Test Accuracy: %s' % mnbsvm.score(X2_test, y2_test))


# ### Classifcation Part 2: Classifying "Earthquake Sensing"  vs. "Earthquake Relevant" Tweets
# 
# In the analysis up until now, we have been able to find and classify tweets which are *earthquake-relevant* vs *earthquake-irrelevant* tweets. However that is not enough to be able to detect earthquakes real-time. Many tweets that are *earthquake-relevant* are tweets in the aftermath of an earthquake when the occurence has been detected much before by seismic sensors and has been distributed by media houses and the internet. Many *earthquake-relevant* tweets might just be information about earthquakes or conferences and what not. Hence the next crucial step is to classify these 2 classes. 
# 
# *Positive Class - Earthquake Sensing Tweets*
# 
# *Negative Class- (only)Earthquake relevant Tweets*
# 
# **Data Collection**
# 
# For this part of the analysis we did not find any data readily available for the *positive class*. For the *negative class* we could just use the data from the earlier dataset. So after trying and failing to extract data from twitter api and trying to sift through that and tag it, we *manually* scraped data off twitter and recorded it in a csv. **This task required significant effort**. We collected tweets by searching for known earthquake timings and locations and searched twitter for tweets which happened within 2-3 minutes of the earthquake. This would serve as the *positive class* for this training set. The total size of this data is ~ 800 tweets
# 
# **Feature Engineering**
# 
# If one eyeballs tweets from the *positive class* one sees that almost all of them contain the word earthquake. Also most of these tweets are small because when people react in the moment, they typically are too surprised to churn out long tweets. Thus *Feature 1* from previous analysis , i.e length of tweet and position of word "earthquake" are good features. This is what the paper we referred to had also used.
# 
# In addition to those, **we came up with a few features of our own. This is another contribution above and beyond what was reported in the research paper**. 
# 
# 1. If one peruses through the tweets, one sees that the tweets tweeted in the moments the earthquake is happening will almost never have any *url link*. There are 2 reasons. One is that again the user is too startled to cite stuff while experiencing an earthquake. Secondly since the first tweeters are by name the first people to know or experience the earthquake, there is no formal news or information available at that moment to refer to via links. In contrast many tweets which are relevant but not sensing are tweets done in the afterhours of an earthquake. Here many people link to either news articles, videos, other information on the internet. Hence *the presence/absence of url/links* is vital discriminatory information between the 2 classes, thus we make this a feature. We used regex to find http in our tweets.
# 
# 
# 2. Similarly when users are tweeting about an earthquake as it is happening, nobody has any quantitative information regarding the magnitude/strength of the earthquake. Thus tweets would hardly have any numerical information except probably the time of occurence or other qualifiers like which time the earthquake is happening in a few months or years. BUT when tweets are tweeted in the afterhours of the earthquake, most of them contain the magnitude of the earthquake which is common information at that point of time. Hence again *the presence/absence of magnitude* is a pretty strong discriminator between positive and negative clases. Thus we chose this a a feature too. We use regex to find tweets with decimal values in tweets, which we believe (and saw from data) that it was a pretty good surrogate to finding magnitude specifically.
# 
# **Classification and Evaluation**
# 
# For classifiers we tried Support Vector Machines and Random Forests on our 4-feature dataset. Both these algorithms have many hyperparameters to tune and for that we used K-fold cross validation accuracy score. This was performed on only the *training data* (80% of total data) to find the best hyperparameter set. Then the complete training data was trained over this set of hyperparameters. Finally to evaluate performance, we tested the final classifer on our *test data* (20% of total data). Our performance from both classifiers was similar, and the SVM just performed marginally better. So we only included that in our analysis below.
# 
# For evaluation we primarly check accuracy but also report precision, recall and F1-score (balanced f-score). For this classifier recall is more important that precision. This is because the recall controls how early (minimum number of tweets) we are able to raise an alarm given users start tweeting about an earthquake. Thus we want to be able to find maximum out of all such tweets. A bad precision will just lead to false alarms, the cost of which is significantly less than missing out on many tweets in an attempt to be precise. However our classifier performs equally good on all counts with more than 85% accuracy and almost similar (~85%) precision and recall. Thus it will pick 4 out of 5 tweets about an earthquake correctly and efficiently.

# In[8]:

tweet_sensing_data = pd.read_csv('Earthquake_sensing_tweets.csv',header=0,delimiter=',')


# In[9]:

def find_url(tweet):
    urls = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', tweet)
    if len(urls) !=0:
        return 1
    else:
        return 0
    
def has_magnitude(tweet):
    decimal = re.findall("\d+\.\d\s", tweet)
    if len(decimal) != 0:
        return 1
    else:
        return 0


# In[10]:

tweet_sensing_data['has_magn']= tweet_sensing_data.Tweet_Text.apply(has_magnitude)
tweet_sensing_data['has_url']= tweet_sensing_data.Tweet_Text.apply(find_url)

tweet_sensing_data = remove_stopword(tweet_sensing_data)
tweet_sensing_data = stemming (tweet_sensing_data)
tweet_sensing_data = Feature_extraction_A(tweet_sensing_data)

tweet_sensing_data = tweet_sensing_data.drop('nostopword',axis=1)
tweet_sensing_data = tweet_sensing_data.drop('stem',axis=1)
# tweet_sensing_data


# In[11]:

dataArray = tweet_sensing_data.values
X3 = dataArray[:,3:]
X3 = np.array(X3, dtype='float')
y3 = dataArray[:,2]
y3 = np.array(y3, dtype='float')

scaler2 = preprocessing.StandardScaler().fit(X3)
X3 = scaler2.transform(X3)

X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.20, random_state=42)


# In[12]:

C_range = np.logspace(-2, 10, 5, base=2)
gamma_range = np.logspace(-9, 1, 5, base=2)
k_options = ['linear','poly','rbf']
params_grids = dict(gamma=gamma_range, C=C_range, kernel=k_options)
grid2 = GridSearchCV(SVC(), param_grid=params_grids, cv=10)
grid2.fit(X3_train,y3_train)

print grid2.best_params_


# In[13]:

#Classificaiton only with both feature set A from paper and our features.

clf2 =  SVC(kernel=grid2.best_params_['kernel'] ,C=grid2.best_params_['C'], 
             gamma=grid2.best_params_['gamma'])
clf2.fit(X3_train,y3_train)
y3_pred = clf2.predict(X3_test)

sklearn.metrics.accuracy_score(y3_test, y3_pred)
sklearn.metrics.precision_score(y3_test, y3_pred)
sklearn.metrics.recall_score(y3_test, y3_pred)
sklearn.metrics.f1_score(y3_test, y3_pred)


# In[18]:

#Classificaiton only with feature set A from paper
clf2.fit(X3_train[:,:2],y3_train)
y3_pred = clf2.predict(X3_test[:,:2])

sklearn.metrics.accuracy_score(y3_test, y3_pred)
sklearn.metrics.precision_score(y3_test, y3_pred)
sklearn.metrics.recall_score(y3_test, y3_pred)
sklearn.metrics.f1_score(y3_test, y3_pred)


# In[19]:

#Classificaiton with only our features.

clf2.fit(X3_train[:,2:],y3_train)
y3_pred = clf2.predict(X3_test[:,2:])

sklearn.metrics.accuracy_score(y3_test, y3_pred)
sklearn.metrics.precision_score(y3_test, y3_pred)
sklearn.metrics.recall_score(y3_test, y3_pred)
sklearn.metrics.f1_score(y3_test, y3_pred)


# ### So we can see that our features peform better and improve tbe classifier.

# ## Putting it together: Querying twitter real-time and finding "earthquake sensing tweets"

# For querying twitter real time, we are using the python 'tweepy' module. We are extracting 50 live streaming tweets from twitter and saving them in a 'live_data.csv' file for later classificaiton. To this 'live_data.csv' file, we are initially appending a set of 10 tweets, out of which 5 of them indicate a current earthquake and the other 5 of them, though related to earthquake, do not indicate a current earthquake. Our classifier needs to be able to detect the tweets which indicate only a current earthquake.

# In[14]:

#Import the necessary methods from tweepy library
import tweepy
from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import json

#Variables that contains the user credentials to access Twitter API 
access_token = '859531338973622277-fdJp7rien3doiULaof2DwcLwIzngo6k'
access_token_secret = 'h3hAd5kzn9qRngThgQyRm9t2p1ErZH1orpAQ4HA15dlG9'
consumer_key = 'wBJ2csLxjMCz0hRwWF7Pw826z'
consumer_secret ='Au0OUtezSVY5VjKrBo9XTz9HJRHIQw2dJPtAVA4K1qBZgGGfh2'


#This is a basic listener that just prints received tweets to stdout.
class StdOutListener(StreamListener):
    
    def __init__(self, api=None):
        super(StdOutListener, self).__init__()
        self.num_tweets = 0
        self.f = csv.writer(open("live_data.csv", "wb+"))
        self.f.writerow(["Tweet_Text"])
        self.f.writerow(["Are we having an earthquake?"])
        self.f.writerow(["EARTHQUAKE?"])
        self.f.writerow(["It shook like crazy #earthquake"])
        self.f.writerow(["WOAHHHHH that was my first earthquake!!!!!"])
        self.f.writerow(["Is it just me or was that an earthquake?"])
        self.f.writerow(["An earthquake of mag 8.2 shook Delhi yesterday!!"])
        self.f.writerow(["RT biggest earthquake in last ten years!! Mag 9.1 richter reported"])
        self.f.writerow(["Attending an earthquake conference today."])
        self.f.writerow(["Japan has frequent earthquakes."])
        self.f.writerow(["Which is worse? Earthquake of 7.2 or 8.1?"])
        

    def on_data(self, data):
        if self.num_tweets < 50:
#             print self.num_tweets
            tweet_data = json.loads(data)
            if 'text' in tweet_data and tweet_data['lang'] == 'en':
                self.f.writerow([tweet_data['text'].encode('utf-8')])
                self.num_tweets += 1
            return True
        else:
            print('Done extracting')
            return False

    def on_error(self, status):
        print status


def get_tweets():

    #This handles Twitter authetification and the connection to Twitter Streaming API
    l = StdOutListener()
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)
    stream = Stream(auth, l)

    #This line filter Twitter Streams to capture data by the keywords: 'python', 'javascript', 'ruby'
    stream.filter(locations=[-180,-90,180,90])
    
get_tweets()


# In[17]:

live_data = pd.read_csv('live_data.csv',header=0)

print('Tweets that indicate a current earthquake : \n')

for tweet in live_data.values.tolist():
    temp_tweet = tweet[0]
    temp_tweet_vector = vectorizer.transform([temp_tweet])
    if mnbsvm.predict(temp_tweet_vector)[0] == 1:
#         print temp_tweet
        tweet_df = pd.DataFrame([temp_tweet],columns=['Tweet_Text'])
        tweet_df['has_magn']= tweet_df.Tweet_Text.apply(has_magnitude)
        tweet_df['has_url']= tweet_df.Tweet_Text.apply(find_url)

        tweet_df = remove_stopword(tweet_df)
        tweet_df = stemming (tweet_df)
        tweet_df = Feature_extraction_A(tweet_df)

        tweet_df = tweet_df.drop('nostopword',axis=1)
        tweet_df = tweet_df.drop('stem',axis=1)
#         print tweet_df
        tempArray = tweet_df.values
        temp_X = tempArray[:,1:]
        temp_X = np.array(temp_X, dtype='float')

        temp_X = scaler2.transform(temp_X)
        temp_y = clf2.predict(temp_X)
        if temp_y == 1:
            print temp_tweet


# ## Conclusions
# 
# *Summary of tasks*
# 
# 1. Searched for data online and obtained 1 pre-curated dataset of tweets related to earthquakes.
# 
# 2. For doing exactly what the paper had done (and also a useful analysis rather than just classification), manually collected data from twitter. This was painful but adds value to the project.
# 
# 3. Replicated features(A,B,C) and classifier(SVM) of the paper.
# 
# 4. Used NBSVM to obtained much better performance than paper.
# 
# 5. Due to data needs and conceptually different thoughts, we did a 2 step classification unlike the paper.
# 
# 6. Based on data exploration, created 2 new features which increased the performance of classification much more than that of Feature A described in paper.
# 
# 
# *Experience and Learnings*
# 
# 1. In analysis, spending time with data crucial and so is data exploration. Better features might get much better performace than better algorithms.
# 
# 2. Understand data needs much before starting analysis. Figure out automated data capture and if not, manual capture will take much more time than anticipated.
# 
# 3. When doing analysis on Jupyter notebook in a team, set variable and functional convention strongly. Merging takes forever otherwise and lot of code will need to be refactored.

# ## References
# 
# [1] M. Sarah, C. Abdur, H. Gregor, L. Ben, and M. Roger, “Twitter and the Micro-Messaging Revolution,” technical report, O’Reilly Radar, 2008.
# 
# [2] A. Java, X. Song, T. Finin, and B. Tseng, “Why We Twitter: Understanding Microblogging Usage and Communities,” Proc. Ninth WebKDD and First SNA-KDD Workshop Web Mining and Social Network Analysis (WebKDD/SNA-KDD ’07), pp. 56-65, 2007.
# 
# [3] B. Huberman, D. Romero, and F. Wu, “Social Networks that Matter: Twitter Under the Microscope,” ArXiv E-Prints, http://arxiv.org/abs/0812.1045, Dec. 2008.
# 
# [4] H. Kwak, C. Lee, H. Park, and S. Moon, “What is Twitter, A Social Network or A News Media?” Proc. 19th Int’l Conf. World Wide Web (WWW ’10), pp. 591-600, 2010.
# 
# [5] G.L. Danah Boyd and S. Golder, “Tweet, Tweet, Retweet: Conversational Aspects of Retweeting on Twitter,” Proc. 43rd Hawaii Int’l Conf. System Sciences (HICSS-43), 2010.
# 
# [6] A. Tumasjan, T.O. Sprenger, P.G. Sandner, and I.M. Welpe, “Predicting Elections with Twitter: What 140 Characters Reveal About Political Sentiment,” Proc. Fourth Int’l AAAI Conf. Weblogs and Social Media (ICWSM), 2010.
# 
# [7] P. Galagan, “Twitter as a Learning Tool. Really,” ASTD Learning Circuits, p. 13, 2009.
# [8] K. Borau, C. Ullrich, J. Feng, and R. Shen, “Microblogging for Language Learning: Using Twitter to Train Communicative and Cultural Competence,” Proc. Eighth Int’l Conf. Advances in Web Based Learning (ICWL ’09), pp. 78-87, 2009.
# 
# [8] Sakaki, Takeshi, Makoto Okazaki, and Yutaka Matsuo. "Tweet analysis for real-time event detection and earthquake reporting system development." IEEE Transactions on Knowledge and Data Engineering 25.4 (2013): 919-931.

# In[ ]:



