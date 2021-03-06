from pyspark.ml import Pipeline
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.feature import CountVectorizer, Tokenizer, HashingTF, IDF, Word2Vec, VectorAssembler
from pyspark.sql.types import *
from pyspark.sql.functions import udf
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import preprocessing as pp
import pyspark as ps
import sys


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: wordcount  ", file=sys.stderr)
        exit(-1)
        
    input_file_path = sys.argv[2]
    output_file_path = sys.argv[3]
    # Setup Spark Context
    sc = ps.SparkContext(appName="kettle")

    # Create Spark DataFrame Schema
    schema = StructType([
        StructField('event_id', StringType(), False),
        StructField('text', StringType(), False),
        StructField('label', FloatType(), False)])

    # Read CSV
    message_df = spark.read.csv(input_file_path, 
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
    check_tag_udf = udf(pp.check_for_tag, IntegerType())
    remove_tags_udf = udf(pp.remove_tags, StringType())



    # Feature engineering
    filtered_null_df = message_df.filter(~message_df['label'].isNull())
    lower_df = filtered_null_df.withColumn('text', lower_udf(filtered_null_df['text']))
    link_df = lower_df.withColumn('has_link', check_for_link_udf(lower_df['text']))
    no_link_df = link_df.withColumn('text', remove_link_udf(link_df['text']))
    replaced_df = no_link_df.withColumn('text', replace_q_udf(no_link_df['text']))
    no_emojis_df = replaced_df.withColumn('text', drop_emojis_udf(replaced_df['text']))
    has_tag_df = no_emojis_df.withColumn('has_tag', check_tag_udf(no_emojis_df['text']))
    no_tags_df = has_tag_df.withColumn('text', remove_tags_udf(has_tag_df['text']))
    word_count_df = no_tags_df.withColumn('word_count', count_words_udf(no_tags_df['text']))
    split_df = word_count_df.withColumn('words', split_message_udf(word_count_df['text']))
    verb_count_df = split_df.withColumn('verb_count', count_verbs_udf(split_df['words']))
    has_q_df = verb_count_df.withColumn('has_q', check_q_udf(verb_count_df['text']))
    stem_df  = has_q_df.withColumn('words', stem_udf(has_q_df['words']))
    no_dupes_df = stem_df.dropDuplicates(['words'])
    no_emptys_df = no_dupes_df.filter(no_dupes_df['word_count']>1)



    # Split data set
    training_df, testing_df = no_emptys_df.randomSplit([.75, .25])

    # Make Spark ML pipeline using a NaiveBayes classifier (for now)
    hashingTF = HashingTF(inputCol='words', outputCol='word_hash', numFeatures=500)
    idf = IDF(minDocFreq=1, inputCol=hashingTF.getOutputCol(), outputCol='tf-idf')
    va = VectorAssembler(inputCols=['has_link', 'verb_count', 'tf-idf', 'word_count', 'has_q', 'has_tag'])
    mp = MultilayerPerceptronClassifier(featuresCol=va.getOutputCol(), layers=[505, 250, 100, 50, 25, 10, 5, 2])

    # Create param grid
    grid = ParamGridBuilder().addGrid(mp.maxIter, [50,100,200]).addGrid(mp.tol, [.0000001, .000001, .0001, .01]).addGrid(mp.stepSize, [.001, .01, .1]).build()

    evaluator = BinaryClassificationEvaluator(rawPredictionCol='prediction')

    pipeline = Pipeline(stages=[hashingTF, idf, va, mp])

    cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, parallelism=4)

    cvModel = cv.fit(no_emptys_df)

    cvModel.write().save(output_file_path)

    sc.stop()