# -*- coding: utf-8 -*-
"""
[He, Li]
[10006204]
[MMAI]
[2020]
[891]
[June 28, 2020]


Submission to Question [1], Part [1]
"""

# import important libraries
import pandas as pd
import numpy as np
import re
import contractions
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from autocorrect import Speller
from sklearn.metrics import f1_score, accuracy_score

# load test data (LEXICON-BASED APPROACH do not require training)
df = pd.read_csv("sentiment_test.csv")

# Check the balance of the data classes
print(df.Polarity.value_counts())


def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)


def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()
    <>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", sample)


# set spell checker's language to English
spell = Speller(lang='en')

# initialize VADER's Analyzer
analyser = SentimentIntensityAnalyzer()


def my_sentiment_analyzer(documents, thres=0.05, verbose=False):
    """take in a corpus of sentences, for every sentence, pre-process followed by labeling sentiment with VADER,
    and finally set all remaining unsure ones to Polarity of 0"""
    preds = np.zeros(len(documents))

    for i, doc in enumerate(documents):
        # preprocessing - multiple actions (remove URL and replace contractions)
        doc = remove_URL(doc)
        doc = replace_contractions(doc)
        # preprocessing - autocorrect spelling
        doc = spell(doc)
        # preprocessing - remove leading/trailing spaces and merge connecting spaces into one
        doc = doc.strip()
        doc = ' '.join(doc.split())
        # try VADER
        score = analyser.polarity_scores(doc)
        if score['compound'] >= thres:
            preds[i] = 1
        elif score['compound'] <= -thres:
            preds[i] = 0
        else:
            # set all remaining unsure texts to polarity of 0
            preds[i] = 0
            # also print out the unknown text
            if verbose == True:
                print(doc)
                print('\n')
    return preds


# predict and store results
pred = my_sentiment_analyzer(df['Sentence'], verbose=False)

# print f1 score
print(f1_score(df['Polarity'], pred))

# print accuracy score
print(accuracy_score(df['Polarity'], pred))

# save input/ground truth/prediction as one csv
df['prediction'] = pred
df.to_csv('q1_ans.csv', index=False)