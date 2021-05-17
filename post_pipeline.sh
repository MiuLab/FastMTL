export MODEL_NAME=$1
export POSTFIX=$2
export TASK_USE_ABS=$3
export BATCHSIZE=$4
export CUDA=$5
export SUBDATASET_NUM=$6
export TOTAL_EPOCH=5
export HTDOCS=final_submission

#Eval
./run_eval_finetune.sh $MODEL_NAME $POSTFIX $TASK_USE_ABS $BATCHSIZE $TOTAL_EPOCH $CUDA $SUBDATASET_NUM
wait
#GET Best scores of finetune
python3 tools/print_score.py ${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS} --finetune --epoch_num $TOTAL_EPOCH --best_dir
wait
#Predict using the best eval checkpoint
./run_predict_finetune.sh $MODEL_NAME $POSTFIX $TASK_USE_ABS $BATCHSIZE $TOTAL_EPOCH $CUDA $SUBDATASET_NUM
wait
#mv the test files
python3 tools/print_score.py ${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS} --finetune --epoch_num $TOTAL_EPOCH --mv_test
wait
#CP fake test files
cp -r FAKE_TEST3/* ./results/FINETUNE_${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS}_BEST/

#SUBMIT, THE 'Y' means to do finetune
cd submit
./submit.sh $MODEL_NAME ${POSTFIX}_${TASK_USE_ABS} Y
cd ..
cp ./results/FINETUNE_${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS}_BEST/submission.zip $HTDOCS/${MODEL_NAME}_${POSTFIX}_${TASK_USE_ABS}_submission.zip

