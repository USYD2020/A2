from pyspark.sql import SparkSession
# import argparse
from pyspark.sql.functions import explode, broadcast, explode_outer, col, collect_list, size, udf
from pyspark.sql.types import *
from pyspark.sql.window import Window

# config the spark session
spark = SparkSession \
    .builder \
    .appName("COMP5349 A2") \
    .config("spark.sql.shuffle.partitions", 10) \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "2G") \
    .getOrCreate()

# spark.conf.set("spark.eventLog.logBlockUpdates.enabled", True)
# spark.sparkContext.setLogLevel("ERROR")
# spark.conf.set("spark.default.parallelism", 100)

# parser = argparse.ArgumentParser()
# parser.add_argument("--output", help="the output directory", default='out.json')
# parser.add_argument("--dataset", help="dataset used", default='out.json')
# args = parser.parse_args()

file_dict = {'train': 's3://comp5349-2022/train_separate_questions.json',
             'test': 's3://comp5349-2022/test.json',
             'large': 's3://comp5349-2022/CUADv1.json'}

# dataset_used = args.dataset
dataset_used = "test.json"

# read in data
# raw_df = spark.read.json(file_dict.get(dataset_used))
raw_df = spark.read.json(dataset_used)
print("Check Schema of raw dataframe\n")
raw_df.printSchema()

# Retrieve values in 'data' column
data_df = raw_df.select((explode("data").alias('data')))
data_df.show(5)

# Retrieve values in 'title' and 'paragraphs' column
title_df = data_df.select("data.title", "data.paragraphs")
paragraphs_df = title_df.select("title", explode("paragraphs")).toDF("title", "paragraphs")
# paragraphs_df.show(5)

# extract information in the paragraphs columns
context_df = paragraphs_df.select("title", "paragraphs.context", "paragraphs.qas").toDF("title", "context",
                                                                                        "qas").cache()


# context_df.show(5)

# ====================
# 
# Apply user defined function function to get segmented sequences into (start, end, source_text) formats
@udf(returnType=ArrayType(
    StructType([

            StructField('segment_start', IntegerType(), False),
            StructField('segment_end', IntegerType(), False),
            StructField("source_text", StringType(), nullable = True)

    ])))
def segment(context):
    """Segment the context string in contract into sequences of 4096 characters except the last sequence"""
    WINDOW_SIZE = 4096
    STRIDE_SIZE = 2048
    start = 0
    res = []
    context_length = len(context)
    while start < context_length:
        end = start + WINDOW_SIZE
        if start + WINDOW_SIZE > context_length:
            end = context_length
        res.append((start, end, context[start:end]))
        start += STRIDE_SIZE
    return res


# [(title, sequence, question, answer_start, answer_end, label)]
print("get segmented sequences, print schema and first 5 rows:\n")

# ====================
# explode the segmented context into rows
segmented_df = context_df.select("title", explode(segment(col("context"))), "qas") \
    .toDF("title", "context", "qas")
segmented_df.printSchema()
segmented_df.show(5)

# explode values in 'qas' column
exploded_qas_df = segmented_df.select("title", "context", explode("qas")).toDF("title", "context", "qas")

# Apply explode_outer on answers to include empty answers
qas_df = exploded_qas_df.select("title", "context", "qas.id", "qas.is_impossible", "qas.question",
                                explode_outer("qas.answers")) \
    .toDF("title", "context", "id", "is_impossible", "question", "answers")
# qas_df.show(5)

# extract values and rename the columns
paragraphs_df = qas_df.select("title", "context", "id", "question", "is_impossible", "answers.answer_start",
                              "answers.text") \
    .toDF("title", "context", "id", "question", "is_impossible", "answer_start", "text")

paragraphs_df.show(5)


# ====================
# user defiend function to label the sequence with a type, as
# impossible_negative, possible_negative, or positive sample
@udf(returnType=StructType([
        StructField('answer_start', IntegerType(), False),
        StructField('answer_end', IntegerType(), False),
        StructField("label", StringType(), True)
]))
def get_type(seq_context, is_impossible, answer_start, answer_text):
    """
    :param is_impossible: a boolean to represent if an answer is present
    :param seq_context: (source_text, seq_start, seq_end)
    :param answer_start: a number
    :param answer_text: a string
    :return: Type of the sample, a string
    """
    # impossible negative, zero occurrences of a particular category
    if is_impossible or not answer_text or not answer_start:
        return 0, 0, "impossible_N"

    # TODO: careful with Order of udf
    seq_start, seq_end, seq_text = seq_context
    answer_end = answer_start + len(answer_text)
    # possible negative, is_impossible” is set to False, but sequence does not contain part of the answer.
    if seq_start > answer_end or seq_end < answer_start:
        return 0, 0, "possible_N"
    # positive sample that contains part or all of the answers to a question
    else:
        # compute the starting point of answer in seq, use 0 if seq_start > answer_start
        # get max
        answer_start = answer_start - seq_start if answer_start - seq_start > 0 else 0
        # compute the starting point of answer in seq, use shorter end as constrained by seq_end and ans_end
        # get min
        answer_end = answer_end - seq_start if answer_end - seq_start < seq_end - seq_start else seq_end - seq_start
        return answer_start, answer_end, "P"


def get_label(context, is_impossible, answer_start, text):
    if text is None or answer_start is None:
        return 0, 0, "imp_neg"
    text_length = len(text)
    answer_end = answer_start + text_length
    seq_start, seq_end, seq = context
    if seq_start > answer_end or seq_end < answer_start:
        return 0, 0, "pos_neg"
    elif seq_start < answer_start and seq_end > answer_end:
        return answer_start - seq_start, answer_end - seq_start, "pos"
    elif seq_start > answer_start and seq_end > answer_end:
        return 0, answer_end - seq_start, "pos"
    elif seq_start < answer_start and seq_end < answer_end:
        return answer_start - seq_start, seq_end - seq_start, "pos"
    else:
        return 0, seq_end - seq_start, "pos"


# Add type to the res df
type_added_df = paragraphs_df.withColumn("label", get_type(col("context"), col("is_impossible"), col("answer_start"),
                                                           col("text")))
print("Adding impossible_negative, possible_negative, or positive sample type to the df\n")
type_added_df.show(5)

# Select the useful columns
labelled_paragraphs_df = type_added_df.select("context.*", "text", "id", "question", "title", "label.*")
print("Select the useful columns:\n")
labelled_paragraphs_df.show(20)

# TODO: modify the following lines, eg P for pos


# Categorise and split the sample sequences by type
# Multiple actions on the same pos_df DataFrame are expected, so we cache it, ie, saves it to storage level 'MEMORY_AND_DISK'
X = labelled_paragraphs_df
pos_df = X.filter(X.seq_type == "P").cache()
impossible_neg_df = X.filter(X.seq_type == "impossible_N")
possible_neg_df = X.filter(X.seq_type == "possible_N")
pos_df.show(20)

# # Apply anti join to extract unique sequences
# Broadcast variables are used to save the copy of data across all nodes. 
# This variable is cached on all the machines and not sent on machines with tasks.
unique_impossible_neg_df = impossible_neg_df.join(broadcast(pos_df), ["seq_source_text"], "leftanti")
# unique_impossible_neg_df.show(5)
unique_possible_neg_df = possible_neg_df.join(broadcast(pos_df), ["seq_source_text"], "leftanti")
unique_possible_neg_df.show(5)

# # group by id on Pitive sample for the calculation of possible negative
P_count_for_P_neg = pos_df.groupBy("id").count()
P_count_for_P_neg.show(10)

# # group by question on Pitive sample for the calculation of impossible negative
P_count_for_imp_neg = pos_df.groupBy("question").count()
P_count_for_imp_neg.show(10)
# # Apply anti join to extract unique sequences
# Broadcast variables are used to save the copy of data across all nodes.
# This variable is cached on all the machines and not sent on machines with tasks.
unique_impossible_neg_df = impossible_neg_df.join(broadcast(pos_df), ["source"], "leftanti")
# unique_impossible_neg_df.show(5)
unique_possible_neg_df = possible_neg_df.join(broadcast(pos_df), ["source"], "leftanti")

# # group by id on Pitive sample for the calculation of possible negative
P_count_for_P_neg = pos_df.groupBy("id").count()
P_count_for_P_neg.show(20)

# # group by question on Pitive sample for the calculation of impossible negative
P_count_for_imp_neg = pos_df.groupBy("question").count()
P_count_for_imp_neg.show(10)

# ====================
# # explode the non-segmented qas column
exploded_qas_df = context_df.select("title", "context", explode("qas")).toDF("title", "context", "qas")
exploded_qas_df.show(5)

# # extract the information in the qas column
full_df = exploded_qas_df.select("context", col("qas.answers").alias("answers"),
                                 col("qas.id").alias("id"),
                                 col("qas.is_impossible").alias("is_impossible"),
                                 col("qas.question").alias("question"), "title")

# full_df = exploded_qas_df.select("context", "qas.answers","qas.id","qas.is_impossible","qas.question", "title").toDF("title", "context", "qas")
full_df.show(5)

# ====================
# group by question to calculate the number of false 'is_imPsible' question
test_paragraphs_ans_id_full_df = full_df.groupBy("question").agg(
    collect_list('is_impossible').alias("is_impossible")) \
    .withColumn("false_count", size(filter(col("is_impossible"), lambda s: s == False)))
test_paragraphs_ans_id_full_df.show(5)

# ====================
# join the denominator and numerator columns and calculate the average number for further filtering.
ave_column = when(col("false_count") == 0, 0).otherwise(round(col("count") / col("false_count")))
test_ave_result = test_paragraphs_ans_id_full_df.join(P_count_for_imp_neg, "question", "left") \
    .withColumn("ave", ave_column).select("question", "ave")
test_ave_result.show(4)

# join the average number back to the imPsible negative samples
test_matched_imp_neg = unique_possible_neg_df.join(broadcast(test_ave_result), "question")
test_matched_imp_neg.show(5)

# join the Pitive sample number back to the Psible negative samples
test_matched_possible_neg = unique_possible_neg_df.join(broadcast(P_count_for_P_neg), "id")
test_matched_possible_neg.show(5)

# ====================
# window function to filter the possible negative samples
pos_window = Window.partitionBy("id").orderBy("answer_start")
pos_neg_result = test_matched_possible_neg.withColumn("row_count", row_number().over(pos_window)) \
    .filter(col("count") >= col("row_count")) \
    .select("source", "question", "answer_start", "answer_end")
# pos_neg_result.show(5)

# window function to filter the impossible negative samples
pos_ave_window = Window.partitionBy("question").orderBy("answer_start")
imp_neg_result = test_matched_imp_neg.withColumn("row_count", row_number().over(pos_ave_window)) \
    .filter(col("ave") >= col("row_count")) \
    .select("source", "question", "answer_start", "answer_end")
# imp_neg_result.show(5)

# select the expected column from the positive samples.
pos_result = pos_df.select("source", "question", "answer_start", "answer_end")
# pos_result.show(5)

# ====================
# union all the samples
result = pos_result.union(imp_neg_result).union(pos_neg_result)
result.show(5)
result.coalesce(1).write.json("output1")
spark.stop()
