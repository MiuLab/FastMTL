export TASK_NAME=all
export POSTFIX=random10
export BATCHSIZE=32
export USE_PER=10
USE_PER=$((100/$USE_PER))
echo $USE_PER

#need to deal with batch size and steps...
declare -A DATA
DATA=( ['mnli']=$((392702/$USE_PER))
    ['rte']=$((2490/$USE_PER))
    ['qqp']=$((363849/$USE_PER)) 
    ['qnli']=$((104743/$USE_PER)) 
    ['mrpc']=$((3668/$USE_PER)) 
    ['sst2']=$((67349/$USE_PER)) 
    ['cola']=$((8551/$USE_PER)) 
    ['stsb']=$((5749/$USE_PER)) 
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
  --seed 13 \
  --per_device_train_batch_size $BATCHSIZE \
  --per_device_eval_batch_size 128 \
  --save_steps $SAVE_STEPS \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --use_data_percent $USE_PER \
  --output_dir ./results/${TASK_NAME}_${POSTFIX}/

#Need to do this incase that the trainer load the state in finetuning for continue training
#When continue training, modify the trainer_log.json back to trainer_state.json, and the model can continue training
mv ./results/${TASK_NAME}_${POSTFIX}/trainer_state.json ./results/${TASK_NAME}_${POSTFIX}/trainer_log.json
