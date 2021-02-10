export MODEL_NAME=$1
export POSTFIX=$2
export BATCHSIZE=$3
export CUDA=$4

#ALL_TASK_NAMES=("mnli" "rte" "qqp" "qnli" "mrpc" "sst2" "cola" "stsb")
ALL_TASK_NAMES=("rte")
#ALL_TASK_NAMES=("rte" "qnli" "mrpc" "sst2" "cola" "stsb")
for i in "${ALL_TASK_NAMES[@]}"
do
    ./run_finetune.sh $i $MODEL_NAME $POSTFIX $BATCHSIZE $CUDA
    wait
done
