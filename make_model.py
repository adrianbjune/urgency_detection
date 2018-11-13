from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.feature import CountVectorizer, Tokenizer, HashingTF, IDF, Word2Vec, VectorAssembler
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import preprocessing as pp
import pyspark as ps

# Setup Spark Context
spark = (
        ps.sql.SparkSession.builder 
        .master("local[4]") 
        .appName("kettle") 
        .getOrCreate()
        )

sc = spark.sparkContext

# Create Spark DataFrame Schema
schema = StructType([
    StructField('event_id', StringType(), False),
    StructField('text', StringType(), False),
    StructField('label', IntegerType(), False)])

# Read CSV
message_df = spark.read.csv('data/labelled_sample_messages.csv', 
                         header=True,
                         schema=schema)

# Create UDFs for Spark
lower_udf = udf(pp.lowercase, StringType())
replace_q_udf = udf(pp.replace_qs, StringType())
stem_udf = udf(pp.stemmer, ArrayType(StringType()))
count_words_udf = udf(pp.count_words, IntegerType())
count_verbs_udf = udf(pp.count_verbs, IntegerType())
check_for_link_udf = udf(pp.check_for_link, IntegerType())
remove_link_udf = udf(pp.remove_link, StringType())
split_message_udf = udf(pp.split_message, ArrayType(StringType()))
check_q_udf = udf(pp.check_for_q, IntegerType())
drop_emojis_udf = udf(pp.drop_emojis, StringType())



# Feature engineering
lower_df = message_df.withColumn('text', lower_udf(message_df['text']))
link_df = lower_df.withColumn('has_link', check_for_link_udf(lower_df['text']))
no_link_df = link_df.withColumn('text', remove_link_udf(link_df['text']))
replaced_df = no_link_df.withColumn('text', replace_q_udf(no_link_df['text']))
no_emojis_df = replaced_df.withColumn('text', drop_emojis_udf(replaced_df['text']))
word_count_df = no_emojis_df.withColumn('word_count', count_words_udf(no_emojis_df['text']))
split_df = word_count_df.withColumn('words', split_message_udf(word_count_df['text']))
verb_count_df = split_df.withColumn('verb_count', count_verbs_udf(split_df['words']))
has_q_df = verb_count_df.withColumn('has_q', check_q_udf(verb_count_df['text']))
stem_df  = has_q_df.withColumn('words', stem_udf(has_q_df['words']))
no_dupes_df = stem_df.dropDuplicates(['words'])
no_emptys_df = no_dupes_df.filter(no_dupes_df['word_count']>1)


# Split data set
training_df, testing_df = no_emptys_df.randomSplit([.75, .25])

# Make Spark ML pipeline using a NaiveBayes classifier (for now)
hashingTF = HashingTF(inputCol='words', outputCol='word_hash')
idf = IDF(minDocFreq=1, inputCol=hashingTF.getOutputCol(), outputCol='tf-idf')
va = VectorAssembler(inputCols=['has_link', 'word_count', 'verb_count', 'tf-idf'])
nb = NaiveBayes(featuresCol=va.getOutputCol())

pipeline = Pipeline(stages=[hashingTF, idf, va, nb])


# Fit model on training data
model = pipeline.fit(training_df)

# Predict on test data
results = model.transform(testing_df)
predictions = results.select('text', 'label', 'prediction', 'probability')

