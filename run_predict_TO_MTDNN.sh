export MODEL_NAME=$1
export POSTFIX=$2
export TASK_NAME=$3
export TO_DIR=$4
export CUDA=$5

CUDA_VISIBLE_DEVICES=$CUDA python3 run_glue.py \
  --model_name_or_path ./results/${MODEL_NAME}_${POSTFIX}/ \
  --task_name $TASK_NAME \
  --do_predict \
  --max_seq_length 128 \
  --per_device_eval_batch_size 512 \
  --output_dir ./results/${MODEL_NAME}_${POSTFIX}/

if [ "$TASK_NAME" = "mnli" ]
then
    cp -r results/${MODEL_NAME}_${POSTFIX}/eval_results_${TASK_NAME}-mm.txt $TO_DIR
    cp -r results/${MODEL_NAME}_${POSTFIX}/test_results_${TASK_NAME}-mm.txt $TO_DIR
fi
cp -r results/${MODEL_NAME}_${POSTFIX}/eval_results_${TASK_NAME}.txt $TO_DIR
cp -r results/${MODEL_NAME}_${POSTFIX}/test_results_${TASK_NAME}.txt $TO_DIR
