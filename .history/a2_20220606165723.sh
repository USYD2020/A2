spark-submit \
    --master yarn \
    --deploy-mode client \
    a2.py \
    --output $1 \
    --dataset $2