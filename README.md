# Twitter-Sentiment-Classifier
A twitter sentiment classifier based on Support Vector Machines algorithm

## Overview

Sentiment analysis is a field of study which identifies the opinion of people expressed in a text using natural language processing tools (https://en.wikipedia.org/wiki/Sentiment_analysis). Social media such as Twitter provides a constant source of textual data, many with an opinion, which can be analyzed using Sentiment Analysis tools.

Billion Object Platform(BOP) aims at developing a platform to allow scholars to interactively explore a billion geotweets and visualizing them on a map. One of the essential components of the BOP pipeline is to analyze the sentiment of the incoming tweet, for which the following sentiment classifier is built.

The code is written in Python and uses scikit-learn library (http://scikit-learn.org/stable/). We use Support Vector Machine (SVM) with Linear kernel (http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#).

## Classes of Sentiment

Two classes: Negative(0) and Positive(1).

## Training Corpus

The following training corpus are publicly available and are collectively used for training the classifier:

#### Stanford Sentiment140

http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip Size: 1,600,000 tweets

#### Polarity Dataset v2.0

http://www.cs.cornell.edu/people/pabo/movie-review-data/ Size: 1000 positive and 1000 negative processed movie reviews

#### Univeristy of Michigan

https://inclass.kaggle.com/c/si650winter11 Size: 7086 sentences from social media (not necessarily twitter)

## Preprocessing the tweet

We have applied a set of pre-processing steps to make tweets suitable for SVM algorithm and improve performance. The following pre-processing has been done on the tweets:

i. Lower Case - Convert the tweets to lower case

ii. URLs - Convert www.* or https?://* to 'URL'

iii. @username - Convert username to '__HANDLE'

iv. #hashtag - Hash tags can give us some useful information, so we replace them with the exact same word without the hash. E.g. #Apple replaced with 'Apple'

v. Trimming the tweet

vi. Repeating words: People often use repeating characters while using colloquial language, such as "I’m happyyyyy". We replace characters repeating more than twice with just two characters, so that the result for above would be "I'm happyy"

vii. Emoticons: Use of emoticons is prevalent in tweets. We identify a set of emoticons and replace them with the reprentative sentiment i.e. '__positive__' or '__negative__'. E.g. ':)' is replaced by '__positive__'. Further, if emoticon(s) are found in the tweet, then the SVM classifier is not called and the tweet is classified as positive or negative simply based on the emoticon

## Stemming

Stemming algorithms are used to find the “root word” or stem of a given word. We have used the Porter Stemmer.

## Tuning of Parameters

Tuning of parameters was done to improve the performance of the SVM classifier. The following parameters are found to give the best results on the cross validation set (20% of the Training Corpus) without compromising much on the speed.

i. TfidfVectorizer:

         min_df=5, 
         max_df=0.95, 
         sublinear_tf = True,
         use_idf = True,
         ngram_range=(1, 2)

ii. Linear SVC:

         C=0.1


## Testing Results

The algorithm achieves an overall precision, recall and f1-score of 0.82 (82%). The details can be found in table below (can be reproduced by running training.py):

                  precision  recall   f1-score  support

          0         0.82      0.80      0.81    160170
          1         0.81      0.83      0.82    161648

         avg/total  0.82      0.82      0.82    321818


## How to use the Classifier

The classifier works for python 3.5.1. Follow the steps below to run it:

#### Required libraries

To use the classifier, you must have the following libraries installed:

i. scikit-learn version is 0.17.1. (http://scikit-learn.org/stable/install.html)

ii. NLTK version 3.2.1 (https://pypi.python.org/pypi/nltk)

iii. numpy (http://www.numpy.org/)

iv. scipy (https://www.scipy.org/)

#### Steps

The classifier has been trained and pickled as svmClassifier.pkl. There is no need to run the training again. However, in future the classifier can be re-trained and tested using training.py in src folder.

i. Download the classifer pipeline: svmClassifier.pkl (https://www.dropbox.com/s/mhxzo8koeby3zj8/svmClassifier.pkl?dl=0) and keep it in same folder as sentiment.py

ii. Download sentiment.py from src folder

iii. Run sentiment.py from the terminal using:

             python3 sentiment.py 

When sentiment.py is called the classifer pipeline: svmClassifier.pkl is loaded in the memory (which takes about 25 secs). Once the classifier is loaded (done only once in the begining), you can input tweet (one at a time) and get its sentiment prediction. The code will continue asking for tweets unless the user enters 'quit'. See illustration of output below:

             Loading the classifier, please wait....
             Classifier loaded, Enter the tweets one at a time or enter quit to escape.
             @Tim I am happy knowing it :) (#User inputs first tweet)
             1                              (#Output of first tweet)
             @NASA I love Science!         (#User inputs second tweet)
             1                               (#Output of second tweet)
             Today was a bad day :( #BadDay     (#User inputs third tweet)
             0                                  (Output of third tweet)
             .  (so on)
             .
             .
             .
             quit (#User inputs quit..program ends)
             >>>


#### Output

The output of the sentiment analyzer is either 0 (Negative) or 1 (Positive). 

Cheers!

