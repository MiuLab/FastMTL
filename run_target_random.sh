export TASK_NAME=all
export POSTFIX=rand500_rand
export BATCHSIZE=32
export USE_PER=100
export CUDA=0
export TASK_USE_ABS=1000
export TASK_LOAD_RANK_DIR="rank_files/${POSTFIX}"
export TASK_RANK_TYPE="random"
export TASK_TRAIN_TASK_DISC="False"
export TASK_DO_PREDICT_TASK="False"
export TASK_TRAIN_DISC_CE="False"
export TASK_TO_VIS_HIDDEN="False"
ALL_TASK_NAMES=("rte" "qnli" "mrpc" "sst2" "cola" "stsb" "mnli" "qqp")
for TASK_MTDNN_TARGET_TASK in "${ALL_TASK_NAMES[@]}"
do
    export TASK_POSTFIX=${POSTFIX}_${TASK_MTDNN_TARGET_TASK}${TASK_USE_ABS}
    # task mtdnn
    ./run_train.sh $TASK_NAME $TASK_POSTFIX $BATCHSIZE $USE_PER $TASK_USE_ABS $TASK_LOAD_RANK_DIR $TASK_RANK_TYPE $CUDA $TASK_TRAIN_TASK_DISC $TASK_MTDNN_TARGET_TASK $TASK_DO_PREDICT_TASK $TASK_TRAIN_DISC_CE $TASK_TO_VIS_HIDDEN
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
#./run_all_finetune.sh $TASK_NAME $POSTFIX $BATCHSIZE $CUDA
#wait
./post_pipeline.sh $TASK_NAME $POSTFIX $TASK_USE_ABS $BATCHSIZE $CUDA
