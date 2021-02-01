export MODEL_NAME=$1
export POSTFIX=$2
export TASK_NAME=all
export CUDA=$3

CUDA_VISIBLE_DEVICES=$CUDA python3 run_glue.py \
  --model_name_or_path ./results/${MODEL_NAME}_${POSTFIX}/ \
  --task_name $TASK_NAME \
  --do_predict \
  --max_seq_length 128 \
  --per_device_eval_batch_size 128 \
  --output_dir ./results/${MODEL_NAME}_${POSTFIX}/
