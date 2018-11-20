import tensorflow as tf
import numpy as np
import preprocessing as pp
import tensorflow_hub as hub

tf.logging.set_verbosity(tf.logging.INFO)

CSV_COLUMNS = ['message_id','text','label']
LABEL_COLUMN = 'label'
DEFAULTS = [['na'], [''], [0]]

INPUT_COLUMNS = [
    tf.feature_column.numeric_column('text'), # Possibly the wrong column type?
    tf.feature_column.numeric_column('link'),
    tf.feature_column.numeric_column('tag'),
    tf.feature_column.numeric_column('question'),
    tf.feature_column.numeric_column('word_count'),
    tf.feature_column.numeric_column('verb_count')
]


def read_dataset(filename, mode, batch_size=256):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return add_engineered(features), label
        
        filename_tf = tf.constant(filename)
        
        dataset = tf.data.TextLineDataset(filename_tf).map(decode_csv)
        
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None
            dataset = dataset.shuddle(buffer_size = 10*batch_size)
        else:
            num_epochs = 1
        
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        batch_features, batch_labels = dataset.make_one_shot_iterator().get_next()
        return batch_features, batch_labels
    return _input_fn
        
        
def add_engineered(features):
    # Get plain text
    plain_text = features['text']
    plain_text = tf.map_fn(pp.replace_qs, plain_text)
    
    features['link'] = tf.map_fn(pp.check_for_link, plain_text, dtype=tf.int32)
    features['tag'] = tf.map_fn(pp.check_for_link, plain_text, dtype=tf.int32)
    features['question'] = tf.map_fn(pp.check_for_q, plain_text, dtype=tf.int32)
    
    plain_text = tf.map_fn(pp.remove_link, plain_text)
    plain_text = tf.map_fn(pp.remove_tags, plain_text)
    plain_text = tf.map_fn(pp.drop_emojis, plain_text)
    
    features['word_count'] = tf.map_fn(pp.count_words, plain_text, dtype=tf.int32)
    features['verb_count'] = tf.map_fn(pp.count_verbs, plain_text, dtype=tf.int32)
    
    module = hub.Module('https://tfhub.dev/google/nnlm-en-dim128/1')
    features['text'] = module(plain_text)
    return features


def serving_input_fn():
    feature_placeholder = {'text':tf.placeholder(tf.string, [None])}
    
    features = add_engineered(feature_placeholder.copy())
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholder)
    
    
def build_estimator(model_dir, hidden_units):
    # Input columns
    (text, link, tag, question, word_count, verb_count) = INPUT_COLUMNS
    
    features = [embedded_text, link, tag, question, word_count, verb_count]
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
    estimator = build_estimator(args['output_dir'], args['hidden_units'].split(' '))
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset(
            filename = args['train_data_path'],
            mode = tf.estimator.ModeKeys.TRAIN,
            batch_size = args['train_batch_size']),
        max_steps = args['train_steps'])
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset(
            filename = args['eval_data_path'],
            mode = tf.estimator.ModeKeys.EVAL,
            batch_size = args['eval_batch_size']),
        steps = 100,
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)