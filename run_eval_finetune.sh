export MODEL_NAME=$1
export POSTFIX=$2
export TASK_POSTFIX=$3
export BATCHSIZE=$4
export TOTAL_EPOCH=$5
export CUDA=$6
export USE_PER=100
USE_PER_DIV=$((100/$USE_PER))

#need to deal with batch size and steps...
declare -A DATA
DATA=( ['mnli']=$((392702/$USE_PER_DIV))
    ['rte']=$((2490/$USE_PER_DIV))
    ['qqp']=$((363849/$USE_PER_DIV)) 
    ['qnli']=$((104743/$USE_PER_DIV)) 
    ['mrpc']=$((3668/$USE_PER_DIV)) 
    ['sst2']=$((67349/$USE_PER_DIV)) 
    ['cola']=$((8551/$USE_PER_DIV)) 
    ['stsb']=$((5749/$USE_PER_DIV)))

ceildiv(){ echo $((($1+$2-1)/$2)); }

echo ${DATA['rte']}
exit

for TASK_NAME in "${!DATA[@]}"
do
    D_NUM=${DATA[$TASK_NAME]}
    SAVE_STEPS=$( ceildiv $D_NUM $BATCHSIZE )
    for epoch in $(seq 1 $(($TOTAL_EPOCH-1)))
    do
        check_steps=$(($epoch * $SAVE_STEPS))
        CUDA_VISIBLE_DEVICES=$CUDA python3 run_glue.py \
          --model_name_or_path ./results/finetune_${MODEL_NAME}_${TASK_POSTFIX}_${TASK_NAME}/checkpoint-${check_steps}/ \
          --task_name $TASK_NAME \
          --do_eval \
          --fp16 \
          --max_seq_length 128 \
          --per_device_eval_batch_size 512 \
          --output_dir ./results/FINETUNE_${MODEL_NAME}_${POSTFIX}_E${epoch}/
    done
    wait
    CUDA_VISIBLE_DEVICES=$CUDA python3 run_glue.py \
      --model_name_or_path ./results/finetune_${MODEL_NAME}_${POSTFIX}_${TASK_NAME}_${POSTFIX}/ \
      --task_name $TASK_NAME \
      --do_eval \
      --fp16 \
      --max_seq_length 128 \
      --per_device_eval_batch_size 512 \
      --output_dir ./results/FINETUNE_${MODEL_NAME}_${POSTFIX}_E${TOTAL_EPOCH}/
done
exit


