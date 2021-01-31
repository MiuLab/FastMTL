MODEL_NAME=all
POSTFIX=0
ALL_TASK_NAMES=("mnli" "rte" "qqp" "qnli" "mrpc" "sst2" "cola" "stsb")

for i in "${ALL_TASK_NAMES[@]}"
do
    dir=results/finetune_${MODEL_NAME}_${POSTFIX}_${i}_$POSTFIX
    zip -rm $dir/CHECKPOINTS.zip  $dir/checkpoint*
    wait
done

