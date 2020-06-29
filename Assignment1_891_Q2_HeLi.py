# -*- coding: utf-8 -*-
"""
[He, Li]
[10006204]
[MMAI]
[2020]
[891]
[June 28, 2020]


Submission to Question [2], Part [1]
"""

# import important libraries
import pandas as pd
import numpy as np
import re, unicodedata
import nltk
import contractions
import inflect
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from autocorrect import Speller
spell = Speller(lang='en')  # set spell checker's language to English
from sklearn.metrics import f1_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# load train & test data
df_train = pd.read_csv("sentiment_train.csv")
df_test = pd.read_csv("sentiment_test.csv")

# combine both train and test data sets into one
df_combined = pd.concat([df_train, df_test])
df_combined.reset_index(inplace=True, drop=True)

# Check the balance of the training data classes
print(df_train.Polarity.value_counts())

def replace_contractions(text):
    """Replace contractions in string of text"""
    return contractions.fix(text)

def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()
    <>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', "", sample)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = ' '.join(words)
    return words

def preprocess(sample):
    sample = remove_URL(sample)
    sample = replace_contractions(sample)
    sample = sample.strip()
    sample = ' '.join(sample.split())
    sample = spell(sample)  # spelling auto correct
    # Tokenize
    words = nltk.word_tokenize(sample)
    # Normalize
    return normalize(words)

# use VADER results as a feature
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
            # set all remaining unsure texts to polarity of 0.5
            preds[i] = 0.5
            # also print out the unknown text
            if verbose:
                print(doc)
                print('\n')
    return preds


# predict and store vader results
vader_predict = my_sentiment_analyzer(df_combined['Sentence'], verbose=False)

# pre-process the sentence column (refer to "preprocess" function above for more details)
df_combined['text_corrected'] = df_combined.Sentence.apply(preprocess)

# Tfidvectorize cleaned up sentences into 500 (max) numeric 2-gram to 10-gram features
# (since VADER already takes care of 1-gram)
from sklearn.feature_extraction.text import TfidfVectorizer
no_features = 500
corpus = df_combined.text_corrected
vectorizer = TfidfVectorizer(min_df=.005, max_df=.5, max_features=no_features, ngram_range=(2, 10))
X = vectorizer.fit_transform(corpus)
df_veced = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names())

# add vader results into training/testing set as a new column
df_veced['vader'] = vader_predict

# perform train/test split
split = 1113 + 1089
X_train = df_veced[:split]
X_test = df_veced[split:]
y_train = df_combined[:split].Polarity
y_test = df_combined[split:].Polarity

# set random seed to 42
random = 42

# use RandomizedSearchCV(CV=20, n_iter=20) to parameter-tune RandomForestClassifier
rfc = RandomForestClassifier()
hyperparams = {'max_features': ['auto', 'log2'],
               'criterion': ['gini', 'entropy'],
               'warm_start': [True, False],
               'n_estimators': [10, 50, 100, 200, 500],
               'max_depth': [None, 5, 10],
               'random_state': [random]}
search = RandomizedSearchCV(rfc,
                            hyperparams,
                            cv=20,
                            scoring='roc_auc',
                            return_train_score=False,
                            error_score=0.0,
                            n_jobs=-1,
                            n_iter=20,
                            random_state=random)
search.fit(X_train, y_train)

# re-train model with the best parameter found above
rfc = search.best_estimator_
rfc.fit(X_train, y_train)

# predict and store results
pred = rfc.predict(X_test)

# print f1 score
print(f1_score(y_test, pred))

# print accuracy score
print(accuracy_score(y_test, pred))

# save input/ground truth/prediction as one csv
df_test['prediction'] = pred
df_test.to_csv('q2_ans.csv', index=False)