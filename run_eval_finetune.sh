export MODEL_NAME=$1
export POSTFIX=$2
export TASK_USE_ABS=$3
export BATCHSIZE=$4
export TOTAL_EPOCH=$5
export CUDA=$6
export SUBDATASET_NUM=$7
export USE_PER=100
USE_PER_DIV=$((100/$USE_PER))

source ./python_alias.sh

#need to deal with batch size and steps...
declare -A DATA
if [ "$SUBDATASET_NUM" = "-1" ]
then
        DATA=(['rte']=$((2490/$USE_PER_DIV))
            ['mrpc']=$((3668/$USE_PER_DIV)) 
            ['stsb']=$((5749/$USE_PER_DIV)) )
else
        DATA=(['rte']=$(($SUBDATASET_NUM/$USE_PER_DIV))
            ['mrpc']=$(($SUBDATASET_NUM/$USE_PER_DIV)) 
            ['stsb']=$(($SUBDATASET_NUM/$USE_PER_DIV)) )
fi

ceildiv(){ echo $((($1+$2-1)/$2)); }

for TASK_NAME in "${!DATA[@]}"
do
    D_NUM=${DATA[$TASK_NAME]}
    export TASK_POSTFIX=${POSTFIX}_${TASK_NAME}${TASK_USE_ABS}
    SAVE_STEPS=$( ceildiv $D_NUM $BATCHSIZE )
    echo "---------------- $SAVE_STEPS"
    for epoch in $(seq 1 $(($TOTAL_EPOCH-1)))
    do
        check_steps=$(($epoch * $SAVE_STEPS))
        CUDA_VISIBLE_DEVICES=$CUDA python3 run_glue.py \
          --model_name_or_path ./results/finetune_${MODEL_NAME}_${TASK_POSTFIX}/checkpoint-${check_steps}/ \
          --task_name $TASK_NAME \
          --do_eval \
          --fp16 \
          --max_seq_length 128 \
          --per_device_eval_batch_size 512 \
          --output_dir ./results/FINETUNE_${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS}_E${epoch}/
    done
    wait
    CUDA_VISIBLE_DEVICES=$CUDA python3 run_glue.py \
      --model_name_or_path ./results/finetune_${MODEL_NAME}_${TASK_POSTFIX}/ \
      --task_name $TASK_NAME \
      --do_eval \
      --fp16 \
      --max_seq_length 128 \
      --per_device_eval_batch_size 512 \
      --output_dir ./results/FINETUNE_${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS}_E${TOTAL_EPOCH}/
done
exit


