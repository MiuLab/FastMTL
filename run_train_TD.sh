export MODEL_NAME=$1
export POSTFIX=$2
export BATCHSIZE=$3
export USE_PER=$4
export USE_ABS=$5
export LOAD_RANK_DIR=$6
export RANK_TYPE=$7
export CUDA=$8
export SHOWNUM=2000
export TRAIN_TASK_DISC=$9
export MTDNN_TARGET_TASK=${10}
export DO_PREDICT_TASK=${11}
export TRAIN_DISC_CE=${12}
export TO_VIS_HIDDEN=${13}
export WEIGHT_LOSS=${14}
export SUBDATASET_FILE=${15}
export SEED=${16}


export TASK_NAME=all

echo $USE_PER
SAVE_STEPS=10000000

if [ "$TRAIN_TASK_DISC" = "True" ]
then
    TRAIN_TASK_DISC="--train_task_disc"
else
    TRAIN_TASK_DISC=""
fi
if [ "$DO_PREDICT_TASK" = "True" ]
then
    DO_PREDICT_TASK="--do_predict_task"
else
    DO_PREDICT_TASK=""
fi
if [ "$TRAIN_DISC_CE" = "True" ]
then
    TRAIN_DISC_CE="--train_disc_CE"
else
    TRAIN_DISC_CE=""
fi
#VIS_HIDDEN
if [ "$TO_VIS_HIDDEN" = "True" ]
then
    VIS_HIDDEN="--vis_hidden --vis_hidden_file figure/${POSTFIX}_S${SHOWNUM} --vis_hidden_num $SHOWNUM"
else
    VIS_HIDDEN=""
fi

if [ "$WEIGHT_LOSS" = "True" ]
then
    WEIGHT_LOSS="--weight_loss"
else
    WEIGHT_LOSS=""
fi

#Run
CUDA_VISIBLE_DEVICES=$CUDA python3 run_glue.py \
  --model_name_or_path bert-base-cased \
  --do_train \
  --do_eval \
  --task_name $TASK_NAME \
  --fp16 \
  --max_seq_length 128 \
  --seed $SEED \
  --per_device_train_batch_size $BATCHSIZE \
  --per_device_eval_batch_size 128 \
  --save_steps $SAVE_STEPS \
  --learning_rate 2e-5 \
  --num_train_epochs 5 \
  --use_data_percent $USE_PER \
  --overwrite_output_dir \
  --use_data_abs $USE_ABS \
  $VIS_HIDDEN \
  --load_rank_dir $LOAD_RANK_DIR \
  --mtdnn_target_task $MTDNN_TARGET_TASK \
  --subdataset_file $SUBDATASET_FILE \
  $DO_PREDICT_TASK \
  $TRAIN_TASK_DISC \
  $TRAIN_DISC_CE \
  $WEIGHT_LOSS \
  --rank_type $RANK_TYPE \
  --output_dir ./results/${MODEL_NAME}_${POSTFIX}/



#Need to do this incase that the trainer load the state in finetuning for continue training
#When continue training, modify the trainer_log.json back to trainer_state.json, and the model can continue training
mv ./results/${MODEL_NAME}_${POSTFIX}/trainer_state.json ./results/${MODEL_NAME}_${POSTFIX}/trainer_log.json
