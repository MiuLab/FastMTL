export TASK_NAME=rte
export POSTFIX=test
export BATCHSIZE=128

#need to deal with batch size and steps...
ARRAY=( "mnli:1000..." )

#For save steps
D_=(${DATA[cow]//:/ })
D=${D[1]}

CUDA_VISIBLE_DEVICES=1 python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --max_seq_length 128 \
  --per_device_train_batch_size $BATCHSIZE \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --output_dir ./results/${TASK_NAME}_${POSTFIX}/

#Need to do this incase that the trainer load the state in finetuning for continue training
#When continue training, modify the trainer_log.json back to trainer_state.json, and the model can continue training
mv ./results/${TASK_NAME}_${POSTFIX}/trainer_state.json ./results/${TASK_NAME}_${POSTFIX}/trainer_log.json
