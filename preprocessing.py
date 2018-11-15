import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# Lemmatizer function 
# Takes in a message as a list of words and returns a list of stemmed words
# LIST -> LIST
def stemmer(message):
    port = PorterStemmer()
    return [port.stem(word) for word in message]

# Function to make everything lowercase
# Takes in a message as a string and returns a string
# STRING -> STRING
def lowercase(message):
    return ' '.join([word.lower() for word in message.split()])

# Function to return word count of the message
# Takes in a message as a string and returns the number of words as an int
# STRING -> INT
def count_words(message):
    return len(message.split())

# Function to replace question marks in between characters with apostrophes
# Takes in a message as a string and returns a string
# Doesn't take into account upper case letters, so use after lowercase
# STRING -> STRING
def replace_qs(message):
    return re.sub(r'[a-z0-9]{1}\?[a-z0-9]{1}', (lambda x: x.group(0)[0]+"'"+x.group(0)[2]), message)

# Function to split a message into a list using NLTK word_tokenize
# Takes in a message as a string and returns a list of words
# STRING -> LIST
def split_message(message):
    return nltk.word_tokenize(message)

# Function to check if a message comtains a link
# Takes in a message as a string and returns a 1 if the message contains a link
# or a 0 if it does not.
# STRING -> INT
def check_for_link(message):
    return int(bool(re.search(r'<http.*>', message)))

# Function to check if a message has a question mark in it
# Takes in a message as a string and returns a 1 if the message contains '?' or
# 0 if it does not.
# STRING -> INT
def check_for_q(message):
    return int(bool(re.search(r'.*\?.*', message)))

# Function to remove Slack emojis from a message. 
# Takes in a message as a string and returns the same message just without any Slack emojis
# STRING -> STRING
def drop_emojis(message):
    return re.sub(r':\S*:', '', message)

# Function to remove links from a message
# Takes in a message as a string and returns a message as a string but without the link
# Returns the message if there is no link
# STRING -> STRING
def remove_link(message):
    return re.sub(r'<http.+>', '', message)

# Function to count how many verbs are in a message
# Takes in a list of words and returns a count of how many verbs are in the message
# LIST -> INT
def count_verbs(message):
    words_pos = nltk.pos_tag(message)
    is_verb = [x[1]=='VB' for x in words_pos]
    return sum(is_verb)

# Function to check if there is a Slack user tag in the message
# Takes in a message as a string and returns 1 if there is a tag and 0 if not
# STRING -> INT
def check_for_tag(message):
    return int(bool(re.search(r'<@\S{9}>', message)))

# Function to remove Slack tags from a message
# Takes in a message as a string and returns the message with any found tags removed.
# STRING -> STRING
def remove_tags(message):
    return re.sub(r'<@\S{9}>', '', message)
