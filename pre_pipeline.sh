export TASK_NAME=all
export POSTFIX=random10
export BATCHSIZE=32
export USE_PER=10

./run_train.sh $TASK_NAME $POSTFIX $BATCHSIZE $USE_PER
wait
./run_all_finetune.sh
wait
./post_pipeline.sh
