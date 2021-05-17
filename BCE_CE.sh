#If want to use full, set this to "-1", else a number
export SUBDATASET_NUM="-1"
#If set to True, Use subdataset in Finetune, too
export SUBDATASET_FINETUNE="False"
# To change seed, check the fast-mtdnn/subdataset_dir for all availabel seed and NUM
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
export WEIGHT_LOSS="False"

source ./python_alias.sh

#export TRAIN_DISC_CE="True"
#export DO_PREDICT_TASK=" "
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
#./run_train.sh $MODEL_NAME $POSTFIX $BATCHSIZE $USE_PER $USE_ABS $LOAD_RANK_DIR $RANK_TYPE $CUDA $TRAIN_TASK_DISC $MTDNN_TARGET_TASK $DO_PREDICT_TASK $TRAIN_DISC_CE $TO_VIS_HIDDEN $WEIGHT_LOSS $SUBDATASET_FILE
#python3 task_disc_to_rank_files.py results/${MODEL_NAME}_${POSTFIX}/
#wait
#reset param


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
    ./run_finetune.sh $TASK_MTDNN_TARGET_TASK $MODEL_NAME $TASK_POSTFIX $BATCHSIZE $CUDA $SUBDATASET_FINETUNE_FILE $SUBDATASET_NUM
    wait
done
#random


./post_pipeline.sh $MODEL_NAME $POSTFIX $TASK_USE_ABS $BATCHSIZE $CUDA $SUBDATASET_NUM
