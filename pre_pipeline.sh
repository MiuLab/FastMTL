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
export TRAIN_TASK_DISC="--train_task_disc"
#export TRAIN_TASK_DISC=""
#export MTDNN_TARGET_TASK="stsb"
export MTDNN_TARGET_TASK="None"
export DO_PREDICT_TASK="--do_predict_task"
export TRAIN_DISC_CE=""
#export TRAIN_DISC_CE="--train_disc_CE"
#export DO_PREDICT_TASK=""

./run_train.sh $TASK_NAME $POSTFIX $BATCHSIZE $USE_PER $USE_ABS $LOAD_RANK_DIR $RANK_TYPE $CUDA $TRAIN_TASK_DISC $MTDNN_TARGET_TASK $DO_PREDICT_TASK $TRAIN_DISC_CE
#wait
#./run_predict.sh $TASK_NAME $POSTFIX $CUDA
#wait
#cd submit
#./submit.sh $TASK_NAME $POSTFIX n
#wait
#cd ..
#./run_all_finetune.sh $TASK_NAME $POSTFIX $BATCHSIZE $CUDA
#wait
#./post_pipeline.sh $TASK_NAME $POSTFIX $BATCHSIZE $CUDA
