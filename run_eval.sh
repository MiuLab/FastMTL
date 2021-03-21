export MODEL_NAME=all
export TASK_NAME=all
export POSTFIX=0

source ./python_alias.sh

CUDA_VISIBLE_DEVICES=1 python3 run_glue.py \
  --model_name_or_path ./results/${MODEL_NAME}_${POSTFIX}/ \
  --task_name $TASK_NAME \
  --do_eval \
  --fp16 \
  --max_seq_length 128 \
  --per_device_eval_batch_size 128 \
  --output_dir ./results/${MODEL_NAME}_${POSTFIX}/
