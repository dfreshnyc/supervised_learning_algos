
# coding: utf-8

# In[432]:

# Function `unpunctuate` takes a string and removes all punctuation

import string

def unpunctuate(s):
    y = "" 
    for x in s:
        if x not in string.punctuation:
            y = y + x
    return y

    
unpunctuate("Hey there! How's it going?")


# In[433]:

# Function get_bag_of_words_for_single_document takes a document or string and returns 
# it's bag of words
import re
from nltk import FreqDist

def get_bag_of_words_for_single_document(s):
    x = unpunctuate(s)
    x = FreqDist(x.split())
    return x.items()
get_bag_of_words_for_single_document("John also likes likes to watch football games.")


# In[434]:

# Function `get_bag_of_words` that uses the above function to achieve the following: 
# Given a list of strings, it returns the total bag of words for all of the documents.


def get_bag_of_words(s):
    y= {}
    ss = ""
    for f in s:
        ss = ss + " " + f
    y = get_bag_of_words_for_single_document(ss)
    return y
get_bag_of_words([
    "John likes to watch movies. Mary likes movies too.",
    "John also likes to watch football games."
])


# In[528]:

# Given a bag of words for all of the documents in our data set 
# Function `turn_words_into_indices` take the keys in the bag of words and alphabetize them

import operator

def turn_words_into_indices(s):
    y = ""
    for x in s:
        y = y + " " + x
    return sorted(get_bag_of_words_for_single_document(y), key=operator.itemgetter(0))
    
print turn_words_into_indices(["John likes to watch movies. Mary likes movies too.",
    "John also likes to watch football games."])


# In[621]:

# Given a document, write a function `vectorize` that turns the document into a list 
# (also will be called a vector) the same length as the number of keys of bag of words where
# for each index of the list will be 1 only if the word at that index in the word list is contained 
# in the document and 0 otherwise.

from sklearn.feature_extraction.text import CountVectorizer

vocab = ["also", "football", "games", "John", "likes", "Mary", "movies", "to", "too", "watch"]
test_set = ["The sun also rises"]

def vectorize(s):
    v = CountVectorizer(vocabulary=vocab, tokenizer=nltk.word_tokenize)
    x = v.fit_transform(s).toarray()
    print x
vectorize(["The sun also rises. Let's go to the movies"])

