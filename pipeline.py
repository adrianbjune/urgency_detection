import pandas as pd
import numpy as np
import preprocessing as pp

from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin

pd.set_option('mode.chained_assignment',None)

class PreprocessingPipe(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        X['text'] = X['text'].apply(pp.replace_qs)
        X['link'] = X['text'].apply(pp.check_for_link)
        X['question'] = X['text'].apply(pp.check_for_q)
        X['tag'] = X['text'].apply(pp.check_for_tag)
        X['text'] = X['text'].apply(pp.drop_emojis)
        X['text'] = X['text'].apply(pp.remove_link)
        X['text'] = X['text'].apply(pp.remove_tags)
        X['split_text'] = X['text'].apply(pp.split_message)
        X['word_count'] = X['text'].apply(pp.count_words)
        X['verb_count'] = X['split_text'].apply(pp.count_verbs)
        
        return X
    
    
class AddTfidf(BaseEstimator, TransformerMixin):
    
    def __init__(self, max_features=1000):
        self.max_features = max_features
        
    def get_params(self, **kwargs):
        return {'max_features': self.max_features}
    
    def fit(self, X, y):
        self.tfidf = TfidfVectorizer(strip_accents='unicode', stop_words='english', max_features=self.max_features)
        self.tfidf.fit(X['text'])
        
        return self
    
    def transform(self, X):
        return pd.concat([X, pd.DataFrame(self.tfidf.transform(X['text']).toarray(), index=X.index)], axis=1)
    

class DropColumns(BaseEstimator, TransformerMixin):
    '''
    Return Dataframe with the given column names removed
    
    Parameters:
    ------------------------
    drop_columns : list strings of column names
    '''
    
    
    def __init__(self, drop_columns=[]):
        self.drop_columns = drop_columns
        
    def get_params(self, **kwargs):
        return {'drop_columns': self.drop_columns}
        
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        
        return X.drop(self.drop_columns, axis=1)