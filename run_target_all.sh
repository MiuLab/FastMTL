export TASK_NAME=all
export POSTFIX=rand500DBCE
export BATCHSIZE=32
export USE_PER=100
export USE_ABS=500
#export LOAD_RANK_DIR=rank_files/bert-base/
#export LOAD_RANK_DIR="results/all_T0/task_disc_rank"
export LOAD_RANK_DIR="None"
# eval_loss_rank, entropy rank
export RANK_TYPE="None"
export CUDA=0
export TRAIN_TASK_DISC="True"
#export TRAIN_TASK_DISC="False"
#export MTDNN_TARGET_TASK="stsb"
export MTDNN_TARGET_TASK="None"
export DO_PREDICT_TASK="True"
export TRAIN_DISC_CE="False"
export TO_VIS_HIDDEN="True"
#export TRAIN_DISC_CE="True"
#export DO_PREDICT_TASK=" "

#./run_train.sh $TASK_NAME $POSTFIX $BATCHSIZE $USE_PER $USE_ABS $LOAD_RANK_DIR $RANK_TYPE $CUDA $TRAIN_TASK_DISC $MTDNN_TARGET_TASK $DO_PREDICT_TASK $TRAIN_DISC_CE $TO_VIS_HIDDEN
wait
#reset param


export TASK_USE_ABS=1000
export TASK_LOAD_RANK_DIR="results/${TASK_NAME}_${POSTFIX}/task_disc_rank"
export TASK_TRAIN_TASK_DISC="False"
export TASK_DO_PREDICT_TASK="False"
export TASK_TRAIN_DISC_CE="False"
export TASK_TO_VIS_HIDDEN="False"
ALL_TASK_NAMES=("rte" "qnli" "mrpc" "sst2" "cola" "stsb" "mnli" "qqp")
for TASK_MTDNN_TARGET_TASK in "${ALL_TASK_NAMES[@]}"
do
    export TASK_POSTFIX=${POSTFIX}_${TASK_MTDNN_TARGET_TASK}${TASK_USE_ABS}
    # task mtdnn
    ./run_train.sh $TASK_NAME $TASK_POSTFIX $BATCHSIZE $USE_PER $TASK_USE_ABS $TASK_LOAD_RANK_DIR $RANK_TYPE $CUDA $TASK_TRAIN_TASK_DISC $TASK_MTDNN_TARGET_TASK $TASK_DO_PREDICT_TASK $TASK_TRAIN_DISC_CE $TASK_TO_VIS_HIDDEN
    wait
    #finetune
    ./run_finetune.sh $TASK_MTDNN_TARGET_TASK $TASK_NAME $TASK_POSTFIX $BATCHSIZE $CUDA
    wait
done
#random


#wait
#./run_predict.sh $TASK_NAME $POSTFIX $CUDA
#wait
#cd submit
#./submit.sh $TASK_NAME $POSTFIX n
#wait
#cd ..
#./post_pipeline.sh $TASK_NAME $POSTFIX $TASK_POSTFIX $BATCHSIZE $CUDA
