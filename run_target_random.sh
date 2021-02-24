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
export TASK_WEIGHT_LOSS="False"

ALL_TASK_NAMES=("rte" "qnli" "mrpc" "sst2" "cola" "stsb" "mnli" "qqp")
TO_DIR=results/${TASK_NAME}_${POSTFIX}_${TASK_USE_ABS}
mkdir $TO_DIR
for TASK_MTDNN_TARGET_TASK in "${ALL_TASK_NAMES[@]}"
do
    export TASK_POSTFIX=${POSTFIX}_${TASK_MTDNN_TARGET_TASK}${TASK_USE_ABS}
    # task mtdnn
    ./run_train.sh $TASK_NAME $TASK_POSTFIX $BATCHSIZE $USE_PER $TASK_USE_ABS $TASK_LOAD_RANK_DIR $TASK_RANK_TYPE $CUDA $TASK_TRAIN_TASK_DISC $TASK_MTDNN_TARGET_TASK $TASK_DO_PREDICT_TASK $TASK_TRAIN_DISC_CE $TASK_TO_VIS_HIDDEN $TASK_WEIGHT_LOSS
    wait
    cp results/${TASK_NAME}_${TASK_POSTFIX}/use_data.txt ${TO_DIR}/use_data_${TASK_MTDNN_TARGET_TASK}.txt
    wait
    ./run_predict_TO_MTDNN.sh $TASK_NAME $TASK_POSTFIX $TASK_MTDNN_TARGET_TASK $TO_DIR $CUDA
    wait
    #finetune
    ./run_finetune.sh $TASK_MTDNN_TARGET_TASK $TASK_NAME $TASK_POSTFIX $BATCHSIZE $CUDA
    wait
done
#random


cd submit
./submit.sh $TASK_NAME ${POSTFIX}_${TASK_USE_ABS} n
wait
cd ..
./post_pipeline.sh $TASK_NAME $POSTFIX $TASK_USE_ABS $BATCHSIZE $CUDA
./show_final_score.sh ${TASK_NAME}_${POSTFIX}_${TASK_USE_ABS}
