export TASK_NAME=all
export POSTFIX=0
export BATCHSIZE=32

#need to deal with batch size and steps...
declare -A DATA
DATA=( ['mnli']=392702 
    ['rte']=2490 
    ['qqp']=363849 
    ['qnli']=104743 
    ['mrpc']=3668 
    ['sst2']=67349 
    ['cola']=8551 
    ['stsb']=5749 
    ['all']=0 )

#Data num
#For not all
D_NUM=${DATA[$TASK_NAME]}
ceildiv(){ echo $((($1+$2-1)/$2)); }
SAVE_STEPS=$( ceildiv $D_NUM $BATCHSIZE )
#For all
if [ "$TASK_NAME" = "all" ]
then
    SAVE_STEPS=0
    for i in "${!DATA[@]}"
    do
        d=${DATA[$i]}
        s=$( ceildiv $d $BATCHSIZE )
        SAVE_STEPS=$(( $SAVE_STEPS + $s ))
    done
fi
echo $SAVE_STEPS

#Run
CUDA_VISIBLE_DEVICES=1 python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --fp16 \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCHSIZE \
  --per_device_eval_batch_size 128 \
  --save_steps $SAVE_STEPS \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --output_dir ./results/${TASK_NAME}_${POSTFIX}/

#Need to do this incase that the trainer load the state in finetuning for continue training
#When continue training, modify the trainer_log.json back to trainer_state.json, and the model can continue training
mv ./results/${TASK_NAME}_${POSTFIX}/trainer_state.json ./results/${TASK_NAME}_${POSTFIX}/trainer_log.json
