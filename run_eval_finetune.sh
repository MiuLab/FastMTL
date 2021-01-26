export MODEL_NAME=all
export POSTFIX=0
export BATCHSIZE=32
export TOTAL_EPOCH=5

declare -A DATA
DATA=( ['mnli']=392702 
    ['rte']=2490 
    ['qqp']=363849 
    ['qnli']=104743 
    ['mrpc']=3668 
    ['sst2']=67349 
    ['cola']=8551 
    ['stsb']=5749 )

ceildiv(){ echo $((($1+$2-1)/$2)); }

for TASK_NAME in "${!DATA[@]}"
do
    D_NUM=${DATA[$TASK_NAME]}
    SAVE_STEPS=$( ceildiv $D_NUM $BATCHSIZE )
    for epoch in $(seq 1 $(($TOTAL_EPOCH-1)))
    do
        check_steps=$(($epoch * $SAVE_STEPS))
        CUDA_VISIBLE_DEVICES=1 python3 run_glue.py \
          --model_name_or_path ./results/finetune_${MODEL_NAME}_${POSTFIX}_${TASK_NAME}_${POSTFIX}/checkpoint-${check_steps}/ \
          --task_name $TASK_NAME \
          --do_eval \
          --fp16 \
          --max_seq_length 128 \
          --per_device_eval_batch_size 128 \
          --output_dir ./results/FINETUNE_${MODEL_NAME}_${POSTFIX}_E${epoch}/
    done
    wait
    CUDA_VISIBLE_DEVICES=1 python3 run_glue.py \
      --model_name_or_path ./results/finetune_${MODEL_NAME}_${POSTFIX}_${TASK_NAME}_${POSTFIX}/ \
      --task_name $TASK_NAME \
      --do_eval \
      --fp16 \
      --max_seq_length 128 \
      --per_device_eval_batch_size 128 \
      --output_dir ./results/FINETUNE_${MODEL_NAME}_${POSTFIX}_E${TOTAL_EPOCH}/
done
exit


