import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('punkt')


file_amazon = open("data/Sentiment Labelled Sentences/amazon_cells_labelled.txt","r")
file_yelp = open("data/Sentiment Labelled Sentences/yelp_labelled.txt","r")
file_imdb = open("data/Sentiment Labelled Sentences/imdb_labelled.txt","r")

lines_amazon = [line_amazon.rstrip("\n") for line_amazon in file_amazon]
lines_yelp=[line_yelp.rstrip("\n") for line_yelp in file_yelp]
lines_imdb=[line_imdb.rstrip("\n") for line_imdb in file_imdb]

import re
X_amazon= [i.split('\t', 1)[0] for i in lines_amazon]
Y_amazon= [i.split('\t', 1)[1] for i in lines_amazon]
X_yelp= [i.split('\t', 1)[0] for i in lines_yelp]
Y_yelp= [i.split('\t', 1)[1] for i in lines_yelp]
X_imdb= [i.split('\t', 1)[0] for i in lines_imdb]
Y_imdb= [i.split('\t', 1)[1] for i in lines_imdb]

# Preprocessing - lowercase
def lowercase_process(X):
    for i in range(len(X)):
        X[i]=X[i].lower()
    return X

X_amazon_lower=lowercase_process(X_amazon)
X_yelp_lower=lowercase_process(X_yelp)
X_imdb_lower=lowercase_process(X_imdb)

# Preprocessing - punctuations
import string
def punctuation_process(X):
    translator = str.maketrans({key: None for key in string.punctuation})
    for i in range(len(X)):
        X[i]=X[i].translate(translator)

    return X

X_amazon_punc=punctuation_process(X_amazon_lower)
X_yelp_punc=punctuation_process(X_yelp_lower)
X_imdb_punc=punctuation_process(X_imdb_lower)

# Preprocessing - stop words
stop_words= stop = stopwords.words('english')

def stopwords_process(X):
    for i in range(len(X)):
        word_list=X[i].split(" ")
        filtered_words = []
        for word in word_list:
            if word not in stop_words:
                filtered_words.append(word)
                X[i]=" ".join(filtered_words)
    return X
X_amazon_stop=stopwords_process(X_amazon_punc)
X_yelp_stop=stopwords_process(X_yelp_punc)
X_imdb_stop=stopwords_process(X_imdb_punc)

#Preprocessing - Stemming
from nltk.stem.snowball import SnowballStemmer

def word_stemmer(X):
    stemmer = SnowballStemmer("english")

    for i in range(len(X)):
        X[i]=stemmer.stem(X[i])
    return X

X_amazon_stem = word_stemmer(X_amazon_stop)
X_yelp_stem = word_stemmer(X_yelp_stop)
X_imdb_stem = word_stemmer(X_imdb_stop)

# Preprocessing - Lemmatization of all the words
from nltk.stem import WordNetLemmatizer
def word_lemmatizer_process(X):
    wordnet_lemmatizer = WordNetLemmatizer()
    for i in range(len(X)):
        X[i]=wordnet_lemmatizer.lemmatize(X[i])
    return X

X_amazon_lemm =word_lemmatizer_process(X_amazon_stem)
X_yelp_lemm =word_lemmatizer_process(X_yelp_stem)
X_imdb_lemm =word_lemmatizer_process(X_imdb_stem)


# Preprocessing - clean text from digits and empty text
def remove_redundancy(document):
    clean_doc = []
    for text in document:
        text = text.split(' ')
        if '' in text:
            text.remove('')
        text = [i for i in text if not i.isdigit()]
        text = [i for i in text if len(i) > 1]
        text = " ".join(text)
        clean_doc.append(text)
    return clean_doc

X_amazon_clean = remove_redundancy(X_amazon_lemm)
X_yelp_clean = remove_redundancy(X_yelp_lemm)
X_imdb_clean = remove_redundancy(X_imdb_lemm)
