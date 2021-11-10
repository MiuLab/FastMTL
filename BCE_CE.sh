export SUBDATASET_NUM="-1"
export SUBDATASET_FINETUNE="False"
export SUBDATASET_SEED=""
export TYPE=$1
export SEED=$2
if [ "$SEED" = "0" ]
then
    SEED=""
fi
export DNUM=$3
export POSTFIX=rand500${TYPE}${SEED}
export BATCHSIZE=32
export USE_PER=100
export RANK_TYPE="None"
export CUDA=0

if [ $SUBDATASET_NUM = "-1" ]
then
    echo "--------- Use ALL ----------"
    export MODEL_NAME=all
    export SUBDATASET_FILE="None"
    export SUBDATASET_FINETUNE_FILE="None"
else
    export SUBDATASET_FILE=subdataset_dir/${SUBDATASET_NUM}_${SUBDATASET_SEED}.json
    if [ $SUBDATASET_FINETUNE = "False" ]
    then
        export IF_SF=""
        export SUBDATASET_FINETUNE_FILE="None"
    else
        export IF_SF="SF_"
        export SUBDATASET_FINETUNE_FILE=$SUBDATASET_FILE
    fi
    export MODEL_NAME=${IF_SF}sub_${SUBDATASET_NUM}_${SUBDATASET_SEED}

fi

export TASK_USE_ABS=${DNUM}
export TASK_LOAD_RANK_DIR="results/${MODEL_NAME}_${POSTFIX}/task_disc_rank"
export TASK_TRAIN_TASK_DISC="False"
export TASK_DO_PREDICT_TASK="False"
export TASK_TRAIN_DISC_CE="False"
export TASK_TO_VIS_HIDDEN="False"
export TASK_WEIGHT_LOSS="False"

ALL_TASK_NAMES=("rte" "mrpc" "stsb")
TO_DIR=results/${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS}
mkdir $TO_DIR
for TASK_MTDNN_TARGET_TASK in "${ALL_TASK_NAMES[@]}"
do
    export TASK_POSTFIX=${POSTFIX}_${TASK_MTDNN_TARGET_TASK}${TASK_USE_ABS}
    # task mtdnn
    ./run_train.sh $MODEL_NAME $TASK_POSTFIX $BATCHSIZE $USE_PER $TASK_USE_ABS $TASK_LOAD_RANK_DIR $RANK_TYPE $CUDA $TASK_TRAIN_TASK_DISC $TASK_MTDNN_TARGET_TASK $TASK_DO_PREDICT_TASK $TASK_TRAIN_DISC_CE $TASK_TO_VIS_HIDDEN $TASK_WEIGHT_LOSS $SUBDATASET_FILE
    wait
    cp results/${MODEL_NAME}_${TASK_POSTFIX}/use_data.txt ${TO_DIR}/use_data_${TASK_MTDNN_TARGET_TASK}.txt
    wait
    ./run_predict_TO_MTDNN.sh $MODEL_NAME $TASK_POSTFIX $TASK_MTDNN_TARGET_TASK $TO_DIR $CUDA
    wait
    #finetune
    ./run_finetune.sh $TASK_MTDNN_TARGET_TASK $MODEL_NAME $TASK_POSTFIX $BATCHSIZE $CUDA $SUBDATASET_FINETUNE_FILE
    wait
done
#random


cd submit
./submit.sh $MODEL_NAME ${POSTFIX}_${TASK_USE_ABS} n
wait
cd ..
./post_pipeline.sh $MODEL_NAME $POSTFIX $TASK_USE_ABS $BATCHSIZE $CUDA $SUBDATASET_NUM
./show_final_score.sh ${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS}
