#ALL_TASK_NAMES=("mnli" "rte" "qqp" "qnli" "mrpc" "sst2" "cola" "stsb")
ALL_TASK_NAMES=("rte" "qnli" "mrpc" "sst2" "cola" "stsb")
for i in "${ALL_TASK_NAMES[@]}"
do
    ./run_finetune.sh $i
    wait
done
