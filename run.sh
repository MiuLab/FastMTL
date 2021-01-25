# optimizer = Adadelta

datasets="cola mnli mrpc qnli qqp rte sst2 stsb wnli"
MODEL_NAME='bert-base-cased'


export TASK_NAME=all
export POSTFIX=0
export BATCHSIZE=32

#need to deal with batch size and steps...
declare -A DATA
DATA=( ['mnli']=392702 
    ['rte']=2490 
    ['qqp']=363849 
    ['qnli']=104743 
    ['mrpc']=3668 
    ['sst2']=67349 
    ['cola']=8551 
    ['stsb']=5749 
    ['all']=0 )

#Data num
#For not all
D_NUM=${DATA[$TASK_NAME]}
ceildiv(){ echo $((($1+$2-1)/$2)); }
SAVE_STEPS=$( ceildiv $D_NUM $BATCHSIZE )
#For all
if [ "$TASK_NAME" = "all" ]
then
    SAVE_STEPS=0
    for i in "${!DATA[@]}"
        do
            d=${DATA[$i]}
            s=$( ceildiv $d $BATCHSIZE )
            SAVE_STEPS=$(( $SAVE_STEPS + $s ))
        done
fi

echo $SAVE_STEPS

### train
for i in $datasets; do
    echo "Now Dataset: $i"
    TASK_NAME=$i
    python run_glue.py \
        --model_name_or_path $MODEL_NAME \
        --task_name $TASK_NAME \
        --do_train \
        --do_eval \
        --do_predict \
        --overwrite_output_dir \
        --max_seq_length 128 \
        --per_device_train_batch_size $BATCHSIZE \
        --learning_rate 2e-5 \
        --num_train_epochs 3 \
        --fp16 \
        --save_steps $SAVE_STEPS \
        --output_dir ./results/$TASK_NAME/
done




