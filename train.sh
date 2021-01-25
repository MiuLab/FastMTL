# optimizer = Adadelta
## task should be in [
## cola,mnli,mrpc,qnli,qqp,rte,sst2,stsb,wnli
##

TASK_NAME='cola'
MODEL_NAME='bert-base-cased'

python run_glue.py \
    --model_name_or_path $MODEL_NAME \
    --task_name $TASK_NAME \
    --do_train \
    --max_seq_length 128 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --output_dir ./results/$TASK_NAME/

