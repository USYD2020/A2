spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 4 \
    Comp5349-A2.py \
    --output $1
