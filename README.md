# TweetQuake
Information Retrieval Course Project

Project Name : TweetQuake 

## Introduction and Problem Statement
We propose to model an algorithm to monitor tweets and to detect the occurrence of an earthquake. The real time nature of tweets allow us to investigate the interaction of events such as earthquake in Twitter. A timely warning of such event could give users time to prepare themselves like hiding at a safe location or escaping from buildings. The detection of earthquake is based on the features extracted from tweet such as the keywords in a tweet, the number of words and their context. 

## Related Work
There have been numerous work carried out by examining Twitter such as topic detection using Twitter to detect emerging topics[1], some researchers have analyzed the network structure of Twitter[2]. Apart from academic fields, numerous twitter applications have emerged, some of them provide analysis of tweets for marketing of products (Tweet-tronics).

There have been some research addressing spatial aspects. Backstorm[3] used queries with location to present probabilistic framework for quantifying spatial variation.

## The proposal of Work
We will build a classifier which clarifies that a tweet is truly referring to an actual earthquake occurrence. We plan to use the support vector machine (SVM) algorithm to generate a model to classify tweets automatically into positive and negative categories. 

We prepare three groups of features for each tweet as follows: 

Features A (statistical features) the number of words in a tweet message, and the position of the query word within a tweet. 
Features B (keyword features) the words in a tweet
Features C (word context features) the words before and after the query word.

Based on the location information from tweets, we can also output the geographic location of the earthquake by computing weighted average, as discussed in the paper.

For evaluating our classifier, we plan to report the evaluation metrics - Precision, Recall and F-measure.

## Dataset:
We are planning to use tweets provided from http://crisislex.org/data-collections.html for earthquake related tweet information. 

## Link to youtube video: 
	
