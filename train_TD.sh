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
export POSTFIX=rand500${TYPE}${SEED}
export BATCHSIZE=32
export USE_PER=100
export USE_ABS=500
export LOAD_RANK_DIR="None"
export RANK_TYPE="None"
export CUDA=1
export TRAIN_TASK_DISC="True"
export MTDNN_TARGET_TASK="None"
export DO_PREDICT_TASK="True"
if [ $TYPE = "DBCE" ]
then
    export TRAIN_DISC_CE="False"
fi
if [ $TYPE = "DCE" ]
then
    export TRAIN_DISC_CE="True"
fi
export TO_VIS_HIDDEN="True"
export WEIGHT_LOSS="False"
export SEED_BASE=$3

if [ $SUBDATASET_NUM = "-1" ]
then
    echo "--------- Use ALL ----------"
    export MODEL_NAME=all
    export SUBDATASET_FILE="None"
else
    export SUBDATASET_FILE=subdataset_dir/${SUBDATASET_NUM}_${SUBDATASET_SEED}.json
    if [ $SUBDATASET_FINETUNE = "False" ]
    then
        export IF_SF=""
    else
        export IF_SF="SF_"
    fi
    export MODEL_NAME=${IF_SF}sub_${SUBDATASET_NUM}_${SUBDATASET_SEED}

fi
./run_train_TD.sh $MODEL_NAME $POSTFIX $BATCHSIZE $USE_PER $USE_ABS $LOAD_RANK_DIR $RANK_TYPE $CUDA $TRAIN_TASK_DISC $MTDNN_TARGET_TASK $DO_PREDICT_TASK $TRAIN_DISC_CE $TO_VIS_HIDDEN $WEIGHT_LOSS $SUBDATASET_FILE $(( SEED_BASE + SEED ))
python3 task_disc_to_rank_files.py results/${MODEL_NAME}_${POSTFIX}/
