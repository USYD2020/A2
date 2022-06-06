spark-submit \
    --master yarn \
    --deploy-mode client \
    --num-executors 3 \
    a2.py \
    --output $1 \
    --dataset $2