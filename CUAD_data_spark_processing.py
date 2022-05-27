
from pyspark.sql import SparkSession
from pyspark.sql.functions import explode


def seq_contain_answer(seq_start_index, seq_end_index, answer_start, answer_end):
    if answer_end < seq_start_index or answer_start > seq_end_index:
        return False
    else:
        return True


def extract_samples(contract):
    title = contract[0]
    context = contract[1][0]
    qas = contract[1][1]

    ps_num = 0
    pns_num = 0
    samples = []
    seq_num = int(len(context) / 2048) + 1

    for num in range(seq_num):
        seq_start_index = num * 2048
        seq_end_index = min(seq_start_index + 4096, len(context) - 1)
        sequence = context[seq_start_index: seq_end_index]

        for qa_pair in qas:
            answers = qa_pair[0]
            is_impossible = qa_pair[2]
            question = qa_pair[3]
            temp_ps = {}
            temp_ns = {}

            if is_impossible:
                sample = (title, sequence, question, 0, 0, "INS")
                # ignore overlap negative sample or sequence already mark as positive
                if temp_ps.get(sequence) is None and temp_ns.get(sequence) is None:
                    temp_ns[sequence] = sample

            else:
                for answer in answers:
                    answer_start = answer[0]
                    answer_text = answer[1]
                    answer_end = answer_start + len(answer_text)
                    if seq_contain_answer(seq_start_index, seq_end_index, answer_start, answer_end):
                        sample = (title, sequence, question, answer_start, answer_end, "PS")
                        samples.append(sample)
                        ps_num += 1
                        temp_ps[sequence] = sample
                        # remove negative sample after sequence marks as positive
                        if temp_ns.get(sequence) is not None:
                            if temp_ns.get(sequence)[5] == "PNS":
                                pns_num -= 1
                            temp_ns.pop(sequence)
                    else:
                        # balance pns and ps
                        if pns_num > ps_num:
                            continue
                        else:
                            sample = (title, sequence, question, 0, 0, "PNS")
                            # ignore overlap negative sample or sequence already mark as positive
                            if temp_ps.get(sequence) is None and temp_ns.get(sequence) is None:
                                temp_ns[sequence] = sample
                                pns_num += 1

            # append unique negative samples
            for value in temp_ns.values():
                samples.append(value)

    return samples


def mark_ps(sample):
    title = sample[0]
    question = sample[2]
    sample_type = sample[5]

    if sample_type == "PS":
        new_sample = ((title, question), 1)
    else:
        new_sample = ((title, question), 0)
    return new_sample


def arrange_join(sample):
    question = sample[0]
    q_ps_list = sample[1][1]

    title = sample[1][0][0]
    sequence = sample[1][0][1]
    answer_start = sample[1][0][2]
    answer_end = sample[1][0][3]
    sample_type = sample[1][0][4]

    # All titles corresponding to the question in the current title, ps of the question
    new_sample = (title, (sequence, question, answer_start, answer_end, sample_type, q_ps_list))

    return new_sample


def calculate_avg_ps_num(title, q_ps_list):
    total_ps_num = 0
    titles = set()
    for element in q_ps_list:
        titles.add(element[0])
        if element[0] != title:
            total_ps_num += element[1]
    return int(total_ps_num / len(titles))


def balance_ins_ps(contract):
    current_title = contract[0]
    sample_list = contract[1]
    samples = []
    questions = {}

    for sample in sample_list:
        question = sample[1]
        if questions.get(question) is None:
            avg_ps = calculate_avg_ps_num(current_title, sample[5])
            questions[question] = [avg_ps, 0]

        ins_limit = questions.get(question)[0]
        ins_num = questions.get(question)[1]
        if sample[4] == "INS":
            if ins_num <= ins_limit:
                ins_num += 1
                questions[question] = [avg_ps, ins_num]
            else:
                continue
        new_sample = (sample[0], sample[1], sample[2], sample[3])
        samples.append(new_sample)

    return samples


# Load Json file as data frame

spark = SparkSession \
    .builder \
    .appName("COMP5349 A2 Data Loading Example") \
    .config("spark.sql.shuffle.partitions", 10) \
    .config("spark.executor.cores", "4") \
    .config("spark.executor.memory", "2G") \
    .getOrCreate()

test_data = "test.json"
test_init_df = spark.read.json(test_data)

# check the schema of data frame
print("check the schema of data frame")
#test_init_df.show(3)
#test_init_df.printSchema()

# modified data schema of data frame
print("modified data schema of data frame")
test_data_df = test_init_df.select((explode("data").alias('data')))
test_title_ps_df = test_data_df.select("data.title", "data.paragraphs")
test_paragraph_df = test_title_ps_df.select("title", explode("paragraphs")).toDF("title", "paragraph")
#test_paragraph_df.printSchema()
#test_paragraph_df.show(3)

# segment contract into samples

# [(title, sequence, question, answer_start, answer_end, type)]
print("segment contract into samples")
test_samples_rdd = test_paragraph_df.rdd.flatMap(extract_samples).cache()


# compute positive samples in each contract of each question

# [((title, question), 1)] The number of ps per question in each sequence
# [((title, question), ps_count)] The number of ps in the sequence in each question
# [(question, (title, ps_num))]
# [(question, [(title1, ps_num), (title2, ps_num)...])]
print("compute positive samples in each contract of each question")
test_ps_marked_rdd = test_samples_rdd.map(mark_ps)\
              .reduceByKey(lambda ps_first, ps_sec: ps_first + ps_sec)\
              .map(lambda r : (r[0][1], (r[0][0], r[1])))\
              .groupByKey()

# join test_ps_marked_rdd and test_samples_rdd

# [(question,  (title, sequence, answer_start, answer_end, type))]
# [(question,  (title, sequence, answer_start, answer_end, type),  [(title, ps_num)])]
# [(title, (sequence, question, answer_start, answer_end, sample_type, [(title, ps_num)]))]
# [(title, [(sequence, question, answer_start, answer_end, sample_type, [(title, ps_num)])])]
print("Join")
ins_preprocess_rdd = test_samples_rdd \
                      .map(lambda sample: (sample[2], (sample[0], sample[1], sample[3], sample[4], sample[5]))) \
                      .join(test_ps_marked_rdd)\
                      .map(arrange_join)\
                      .groupByKey()

# filter impossible negative samples and return final result

# [(source, sequence, question, answer_start, answer_end, type)]
print("filter impossible negative samples and return final result")
question_answering_model_rdd = ins_preprocess_rdd.flatMap(balance_ins_ps)

# convert rdd to json file
df = question_answering_model_rdd.toDF(["source", "question", "answer_start", "answer_end"])
df.write.json("/output")




