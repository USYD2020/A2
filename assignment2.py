def segmentation(data):
  context = data[1][2][0]

  WINDOW_SIZE = 4096
  STRIDE = 2048

  length = len(context)
  start = 0

  res = []
  index = []

  while start < length:
    end = start + WINDOW_SIZE
    if end > length:
      res.append(context[start:])
    else:
      res.append(context[start:end])
    index.append((start,end))
    start += STRIDE

"""
  Row(answers=[Row(answer_start=14, text='SUPPLY CONTRACT')], 
  id='LohaCompanyltd_20191209_F-1_EX-10.16_11917878_EX-10.16_Supply Agreement__Document Name', 
  is_impossible=False, 
  question='Highlight the parts (if any) of this contract related to "Document Name" that should be reviewed by a lawyer. Details: The name of the contract') 
"""

def extract_qas(data):
  id = data[0]
  qas_list = data[1][1]

  index = 0
  res = []

  for qas in qas_list:
    answers = qas[0]
    title = qas[1]
    is_impossible = qas[2]
    question = qas[3]


    if is_impossible:
      answer_start = 0
      text = ""
      res.append((index,answer_start,answer_start+len(text),is_impossible,question))
    else:
      for answer in answers:
        answer_start = answer[0]
        text = answer[1]
        res.append((index,answer_start,answer_start+len(text),is_impossible,question))

    index+=1
    #res.append(qas[0])

  return id,res

def cal_positive_sample(data):
  segments = data[1][0]
  qas_list = data[1][1]

  res = []
  positive_count = {}

  for qas in qas_list:
    start = qas[1]
    end = qas[2]
    
    count = 0
    if qas[0] not in positive_count: positive_count[qas[0]] = 0
    if qas[3] == False:
      for range_seg in segments[0]:
        l,r = range_seg
        if (start >= l and start <= r) or (end >= l and end <= r):
          positive_count[qas[0]] += 1
    res.append((qas[0],positive_count[qas[0]]))

  return res

"""
{
source: "... ",
question: "... ",
answer _ start: s,
answer _ end: e
}
"""

def generate_qa_sample(data):
  segments = data[1][0]
  qas_list = data[1][1]

  res = []
  positive_count = {}
  source_set = set()

  # Positive samples
  for qas in qas_list:
    start = qas[1]
    end = qas[2]
    
    count = 0
    if qas[0] not in positive_count: positive_count[qas[0]] = 0
    if qas[3] == False:
      index = 0
      for range_seg in segments[0]:
        l,r = range_seg
        if (start >= l and start <= r) or (end >= l and end <= r):
          positive_count[qas[0]] += 1
          source_set.add(index)
          res.append({"source" : segments[1][index], 
                      "question": qas[4],
                      "answer_start": start,
                      "answer_end": end})
        index += 1

  
  #print(source_set)
  # Impossible negative samples
  for qas in qas_list:
    #print(qas,avg_positive_count[qas[0]])
    if qas[3]:
        count = 0
        for i in range(len(segments[0])):
          if i not in source_set:
            #print(qas[0],"yes")
            if count < avg_positive_count[qas[0]]:
              count += 1
              res.append({"source" : segments[1][i], 
                      "question": qas[4],
                      "answer_start": 0,
                      "answer_end": 0})
         
  # Possible negative samples
  for qas in qas_list:
    if qas[3]:
        count = 0
        for i in range(len(segments[0])):
          if i not in source_set:
            if count < positive_count[qas[0]]:
              count += 1
              res.append({"source" : segments[1][i], 
                      "question": qas[4],
                      "answer_start": 0,
                      "answer_end": 0})
  return res


from pyspark.sql import SparkSession
from pyspark.sql.functions import explode
import pyspark.sql.functions as F

if __name__ == "__main__":

    spark = SparkSession \
    .builder \
    .appName("COMP5349 A2") \
    .getOrCreate()

    raw_df = spark.read.json("s3a://comp5349-assignment-data/assignment2_data/CUADv1.json")
    print(raw_df.printSchema())

    df = raw_df.select("data.title",explode("data.paragraphs").alias("paragraphs"))
    df = df.select("title",explode("paragraphs.qas").alias("qas"),"paragraphs.context")
    df = df.select("*").withColumn("id", F.monotonically_increasing_id())
    print(df.printSchema())

    print(df.show(5))

    rdd = df.rdd.map(list)
    # Index 0: id, Index 1: title, Index 2: qas, index 3: context
    rdd = rdd.map(lambda x: (x[3],(x[0][x[3]],x[1],x[2])))
    print(rdd.take(1))

    segment_rdd = rdd.map(segmentation)
    print(segment_rdd.take(1))

    qas_rdd = rdd.map(extract_qas)
    print(qas_rdd.take(1))

    qa_rdd = segment_rdd.join(qas_rdd)
    print(qa_rdd.take(1))

    positive_count_rdd = qa_rdd.map(cal_positive_sample).flatMap(lambda x:x)
    print(positive_count_rdd.take(10))

    positive_count_rdd_temp = positive_count_rdd.aggregateByKey((0,0), lambda x,y: (x[0] + y,    x[1] + 1),
                                       lambda x,y: (x[0] + y[0], x[1] + y[1]))
    avg_positive_count_rdd = positive_count_rdd_temp.mapValues(lambda x: x[0]/x[1]).map(lambda x: (x[0],int(x[1])+1 if (x[1]-int(x[1])) > 0.5  else int(x[1])))
    avg_positive_count_temp = avg_positive_count_rdd.collect()
    avg_positive_count = {}
    for k,v in avg_positive_count_temp: avg_positive_count[k] = v
    print(avg_positive_count_rdd.take(10))

    qa_samples = qa_rdd.map(generate_qa_sample)
    print(qa_samples.take(1))

    print(qa_samples.flatMap(lambda x:x).take(1))
    print(qa_samples.flatMap(lambda x:x).count())

    results = qa_samples.collect()

    spark.stop()