import pandas as pd
import pyspark as ps
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# Lemmatizer function 
# Takes in a message as a string and returns a string
def stemmer(message):
    port = PorterStemmer()
    return ' '.join([port.stem(word) for word in message.split()])

# Function to make everything lowercase
# Takes in a message as a string and returns a string
def lowercase(message):
    return ' '.join([word.lower() for word in message.split()])

# Function to return word count of the message
# Takes in a message as a string and returns the number of words as an int
def word_count(message):
    return len(message.split())

# Function to replace question marks in between characters with apostrophes
# Takes in a message as a string and returns a string
# Doesn't take into account upper case letters, so use after lowercase
def replace_qs(message):
    return re.sub(r'[a-z0-9]{1}\?[a-z0-9]{1}', (lambda x: x.group(0)[0]+"'"+x.group(0)[2]), message)