import pandas as pd
import numpy as np
import pipeline as pipe
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from imblearn.over_sampling import SMOTE
import pickle

pd.set_option('mode.chained_assignment',None)

data = pd.read_csv('data/5_labelled.csv')
X = data.drop(['label','event_id'], axis=1)
y = data['label']

'''
smote = SMOTE()

X_smote_array, y_smote_array = smote.fit_sample(X, y)
X_smote = ps.DataFrame(data=X_smote_array, columns=['text'])
y_smote = pd.DataFrame(data=y_smote_array, columns=['label'])
'''
pipeline = Pipeline([('pp', pipe.PreprocessingPipe()),
                     ('tfidf', pipe.AddTfidf()),
                     ('drop', pipe.DropColumns(drop_columns=['text', 'split_text'])),
                     ('mlp', MLPClassifier())])


params = {'tfidf__max_features':[500,1000],
          'mlp__hidden_layer_sizes':[[500,200, 100], [500]],
          'mlp__batch_size':[300,500, 800],
          'mlp__activation':['tanh', 'relu'],
          'mlp__solver':['lbfgs', 'sgd', 'adam'],
          'mlp__learning_rate':['adaptive'],
          'mlp__max_iter':[10000]}

model = RandomizedSearchCV(estimator=pipeline, param_distributions=params, 
                           n_iter=30, n_jobs=-1, verbose=4)

model.fit(X, y)

best_model = model.best_estimator_

pickle.dump(model.best_estimator_, open('best_model.pkl', 'wb'))
pickle.dump(model, open('grid_model.pkl', 'wb'))