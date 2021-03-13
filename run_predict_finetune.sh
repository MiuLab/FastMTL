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
#Dict(associate array) don't retain the order
declare -A DATA
if [ "$SUBDATASET_NUM" = "-1" ]
then
        DATA=( ['mnli']=$((392702/$USE_PER_DIV))
            ['rte']=$((2490/$USE_PER_DIV))
            ['qqp']=$((363849/$USE_PER_DIV)) 
            ['qnli']=$((104743/$USE_PER_DIV)) 
            ['mrpc']=$((3668/$USE_PER_DIV)) 
            ['sst2']=$((67349/$USE_PER_DIV)) 
            ['cola']=$((8551/$USE_PER_DIV)) 
            ['stsb']=$((5749/$USE_PER_DIV)) 
            ['all']=0 )
else
        DATA=( ['mnli']=$(($SUBDATASET_NUM/$USE_PER_DIV))
            ['rte']=$(($SUBDATASET_NUM/$USE_PER_DIV))
            ['qqp']=$(($SUBDATASET_NUM/$USE_PER_DIV)) 
            ['qnli']=$(($SUBDATASET_NUM/$USE_PER_DIV)) 
            ['mrpc']=$(($SUBDATASET_NUM/$USE_PER_DIV)) 
            ['sst2']=$(($SUBDATASET_NUM/$USE_PER_DIV)) 
            ['cola']=$(($SUBDATASET_NUM/$USE_PER_DIV)) 
            ['stsb']=$(($SUBDATASET_NUM/$USE_PER_DIV)) 
            ['all']=0 )
fi
#Use this to retain the order
DATA_ORDER=("rte" "mrpc" "stsb")

ceildiv(){ echo $((($1+$2-1)/$2)); }
#For best epoch
BEST_EPOCH_FILE=results/FINETUNE_${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS}_BEST/best_epoch.txt
getArray() {
    best_epoch=() # Create array
    while IFS= read -r line # Read a line
    do
        best_epoch+=("$line") # Append line to the array
    done < "$1"
}

getArray $BEST_EPOCH_FILE
cnt_task=0
for I in "${!DATA_ORDER[@]}"
do
    TASK_NAME=${DATA_ORDER[I]}
    echo $TASK_NAME
    D_NUM=${DATA[$TASK_NAME]}
    export TASK_POSTFIX=${POSTFIX}_${TASK_NAME}${TASK_USE_ABS}
    SAVE_STEPS=$( ceildiv $D_NUM $BATCHSIZE )
    epoch=${best_epoch[cnt_task]}
    echo $TASK_NAME
	if [ "$epoch" != "$TOTAL_EPOCH" ]
	then
        check_steps=$(($epoch * $SAVE_STEPS))
        CUDA_VISIBLE_DEVICES=$CUDA python3 run_glue.py \
          --model_name_or_path ./results/finetune_${MODEL_NAME}_${TASK_POSTFIX}/checkpoint-${check_steps}/ \
          --task_name $TASK_NAME \
          --do_predict \
          --fp16 \
          --max_seq_length 128 \
          --per_device_eval_batch_size 512 \
          --output_dir ./results/FINETUNE_${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS}_E${epoch}/
	else
		CUDA_VISIBLE_DEVICES=$CUDA python3 run_glue.py \
          --model_name_or_path ./results/finetune_${MODEL_NAME}_${TASK_POSTFIX}/ \
          --task_name $TASK_NAME \
          --do_predict \
          --fp16 \
          --max_seq_length 128 \
          --per_device_eval_batch_size 512 \
          --output_dir ./results/FINETUNE_${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS}_E${TOTAL_EPOCH}/
	fi
    wait
    cnt_task=$((cnt_task+1))
done

