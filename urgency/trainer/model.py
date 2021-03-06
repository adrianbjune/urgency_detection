import tensorflow as tf
import numpy as np
import trainer.preprocessing as pp
import tensorflow_hub as hub
import pandas as pd
from google.cloud import storage

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = ['message_id','text','label']
LABEL_COLUMN = 'label'
DEFAULTS = [['na'], [''], [0]]

TF_INPUT_COLUMNS = [
     #hub.text_embedding_column('text', 'https://tfhub.dev/google/nnlm-en-dim128/1'),
     #tf.placeholder(tf.string, [None],'text'), # Possibly the wrong column type?
     tf.feature_column.numeric_column('link', dtype=tf.int64),
     tf.feature_column.numeric_column('tag', dtype=tf.int64),
     tf.feature_column.numeric_column('question', dtype=tf.int64),
     tf.feature_column.numeric_column('word_count', dtype=tf.int64),
     tf.feature_column.numeric_column('verb_count', dtype=tf.int64)
]

INPUT_COLUMNS = ['text', 'link', 'tag', 'question', 'word_count', 'verb_count']

#embedded_text = hub.text_embedding_column('text', 'https://tfhub.dev/google/nnlm-en-dim128/1')

def my_auc(labels, predictions):
    return {'auc': tf.metrics.auc(labels, predictions['class_ids'])}

def read_csv(project, bucket, path):
    client = storage.Client(project=project)
    bucket = client.get_bucket(bucket)
    blob = bucket.get_blob(path)
    
    text_array = blob.download_as_string().decode('utf-8').split('\n')
    ids = []
    messages = []
    labels = []
    for line in text_array:
        comma_split = line.split(',')
        ids.append(comma_split[0])
        labels.append(int(float(comma_split[-1])))
        messages.append(''.join(comma_split[1:-1]))
    df_dict = {'id':ids, 'text':messages, 'label':labels}
    return pd.DataFrame.from_dict(df_dict)

def read_dataset(project, bucket, path, mode, batch_size=256):

    data_pd = read_csv(project, bucket, path)
    dataset = add_engineered(data_pd)
    print(dataset.info())
    
    if mode == tf.estimator.ModeKeys.TRAIN:
#         num_epochs = None
#         dataset = dataset.shuffle(buffer_size = 10*batch_size)
        return tf.estimator.inputs.pandas_input_fn(
            x=dataset[['text', 'link', 'tag', 'question', 'word_count', 'verb_count']], y=dataset['label'],
            num_epochs=None, shuffle=True,
            batch_size=batch_size)
    else:
        return tf.estimator.inputs.pandas_input_fn(
            x=dataset[['text', 'link', 'tag', 'question', 'word_count', 'verb_count']], y=dataset['label'],
            num_epochs=1, shuffle=False,
            batch_size=batch_size)


        
def add_engineered(data_pd):

    data_pd['text'] = data_pd['text'].apply(pp.replace_qs)
    data_pd['link'] = data_pd['text'].apply(pp.check_for_link)
    data_pd['tag'] = data_pd['text'].apply(pp.check_for_tag)
    data_pd['question'] = data_pd['text'].apply(pp.check_for_q)
    data_pd['text'] = data_pd['text'].apply(pp.remove_link)
    data_pd['text'] = data_pd['text'].apply(pp.remove_tags)
    data_pd['text'] = data_pd['text'].apply(pp.drop_emojis)
    data_pd['word_count'] = data_pd['text'].apply(pp.count_words)
    data_pd['split_text'] = data_pd['text'].apply(pp.split_message)
    data_pd['verb_count'] = data_pd['split_text'].apply(pp.count_verbs)
    
    return data_pd
    


def serving_input_fn():
    features = {'text':tf.placeholder(tf.string, [None]),
                'link':tf.placeholder(tf.int64, [None]),
                'tag':tf.placeholder(tf.int64, [None]),
                'question':tf.placeholder(tf.int64, [None]),
                'word_count':tf.placeholder(tf.int64, [None]),
                'verb_count':tf.placeholder(tf.int64, [None])}
    
    return tf.estimator.export.ServingInputReceiver(features, features)
    
    
    
def build_estimator(model_dir, hidden_units):
    # Input columns
    (link, tag, question, word_count, verb_count) = TF_INPUT_COLUMNS
    embedded_text = hub.text_embedding_column('text', 'https://tfhub.dev/google/Wiki-words-500-with-normalization/1')
    features = [embedded_text, link, tag, question, word_count, verb_count]#[text, link, tag, question, word_count, verb_count]
    #print('embedded_text: {}\n'.format(embedded_text))
    print('link: {}\n'.format(link))
    print('tag: {}\n'.format(tag))
    print('question: {}\n'.format(question))
    print('word_count: {}\n'.format(word_count))
    print('verb_count: {}\n'.format(verb_count))
    estimator = tf.estimator.DNNClassifier(
        hidden_units = hidden_units,
        feature_columns = features,
        model_dir = model_dir,
        n_classes = 2,
        optimizer='Adagrad',
        dropout = .5,
    )
    
    # Add AUC score to metrics
    estimator = tf.contrib.estimator.add_metrics(estimator, my_auc)
    return estimator
    

def train_and_evaluate(args):
    print('Building estimator...\n')
    estimator = build_estimator(args['output_dir'], args['hidden_units'].split(' '))
    print('Building training spec...\n')
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(
            project = args['project'],
            bucket = args['bucket'],
            path = args['train_data_path'],
            mode = tf.estimator.ModeKeys.TRAIN,
            batch_size = args['train_batch_size']),
        max_steps = args['train_steps'])
    print('Building exporters...\n')
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    final_exporter = tf.estimator.FinalExporter('final', serving_input_fn)
    print('Building evaluation spec...')
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(
            project = args['project'],
            bucket = args['bucket'],
            path = args['eval_data_path'],
            mode = tf.estimator.ModeKeys.EVAL,
            batch_size = args['eval_batch_size']),
        steps = 100,
        exporters = [exporter,final_exporter],
        start_delay_secs=args['eval_delay_secs'],
        throttle_secs=args['eval_delay_secs'])
    print('Training and evaluating...\n')
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)