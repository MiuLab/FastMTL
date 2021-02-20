export MODEL_NAME=all
export TASK_NAME=all
export POSTFIX=T0
export SHOWNUM=2000

if [ "$SHOWNUM" = "-1" ]
then
    CUDA_VISIBLE_DEVICES=0 python3 run_glue.py \
      --model_name_or_path ./results/${MODEL_NAME}_${POSTFIX}/ \
      --task_name $TASK_NAME \
      --do_eval \
      --fp16 \
      --max_seq_length 128 \
      --per_device_eval_batch_size 256 \
      --vis_hidden \
      --vis_hidden_file figure/$POSTFIX \
      --vis_hidden_num $SHOWNUM \
      --output_dir ./results/${MODEL_NAME}_${POSTFIX}/
else
    CUDA_VISIBLE_DEVICES=0 python3 run_glue.py \
      --model_name_or_path ./results/${MODEL_NAME}_${POSTFIX}/ \
      --task_name $TASK_NAME \
      --do_eval \
      --fp16 \
      --max_seq_length 128 \
      --per_device_eval_batch_size 256 \
      --vis_hidden \
      --vis_hidden_file figure/${POSTFIX}_S${SHOWNUM} \
      --vis_hidden_num $SHOWNUM \
      --output_dir ./results/${MODEL_NAME}_${POSTFIX}/
fi

