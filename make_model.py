from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, NaiveBayes
from pyspark.ml.feature import CountVectorizer, Tokenizer, HashingTF, IDF, Word2Vec
from pyspark.sql.types import *
from pyspark.sql.functions import udf
import preprocessing as pp


# Create Spark DataFrame Schema
schema = StructType([
    StructField('event_id', StringType(), False),
    StructField('text', StringType(), False),
    StructField('label', IntegerType(), False)])

# Read CSV
message_df = spark.read.csv('data/labelled_sample_messages.csv', 
                         header=True,
                         schema=schema)

# Feature engineering
lower_df = message_df.withColumn('text', lower_udf(message_df['text']))
replaced_df = lower_df.withColumn('text', replace_q_udf(df_lower['text']))
final_df  = replaced_df.withColumn('text', stem_udf(df_replaced['text']))

# Split data set
training_df, testing_df = final_df.randomSplit([.75, .25])

# Make Spark ML pipeline using a NaiveBayes classifier (for now)
tokenizer = Tokenizer(inputCol='text', outputCol='words')
hashingTF = HashingTF(inputCol=tokenizer.getOutputCol(), outputCol='features')
idf = IDF(minDocFreq=1, inputCol=hashingTF.getOutputCol(), outputCol='idf')
nb = NaiveBayes()

pipeline = Pipeline(stages=[tokenizer, hashingTF, idf, nb])


# Fit model on training data
model = pipeline.fit(training_df)

# Predict on test data
results = model.transform(testing_df)
predictions = results.select('text', 'label', 'prediction', 'probability')

