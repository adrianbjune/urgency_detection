from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.feature import CountVectorizer, Tokenizer, HashingTF, IDF, Word2Vec
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
stem_udf = udf(pp.stemmer, StringType()) 
word_count_udf = udf(pp.word_count, IntegerType())


# Feature engineering
lower_df = message_df.withColumn('text', lower_udf(message_df['text']))
replaced_df = lower_df.withColumn('text', replace_q_udf(lower_df['text']))
final_df  = replaced_df.withColumn('text', stem_udf(replaced_df['text']))

# Split data set
training_df, testing_df = final_df.randomSplit([.75, .25])

# Make Spark ML pipeline using a NaiveBayes classifier (for now)
tokenizer = Tokenizer(inputCol='text', outputCol='words')
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='features')
idf = IDF(minDocFreq=1, inputCol=hashingTF.getOutputCol(), outputCol='tf-idf')
nb = NaiveBayes(featuresCol=idf.getOutputCol())

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb])


# Fit model on training data
model = pipeline.fit(training_df)

# Predict on test data
results = model.transform(testing_df)
predictions = results.select('text', 'label', 'prediction', 'probability')

