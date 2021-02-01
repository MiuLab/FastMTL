export TASK_NAME=all
export POSTFIX=mlm_loss10
export BATCHSIZE=32
export USE_PER=10
export LOAD_RANK_DIR=rank_files/bert-base/
# eval_loss_rank, entropy rank
export RANK_TYPE=eval_loss_rank
export CUDA=1

./run_train.sh $TASK_NAME $POSTFIX $BATCHSIZE $USE_PER $LOAD_RANK_DIR $RANK_TYPE $CUDA
wait
./run_predict.sh $TASK_NAME $POSTFIX $CUDA
wait
cd submit
./submit.sh $TASK_NAME $POSTFIX n
wait
cd ..
./run_all_finetune.sh $TASK_NAME $POSTFIX $BATCHSIZE $CUDA
wait
./post_pipeline.sh $TASK_NAME $POSTFIX $BATCHSIZE $CUDA
