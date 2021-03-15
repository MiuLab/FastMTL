#Use this script to print results, the example is for TD-MTDNN(500) and TO-MTDNN(1000)
#MODEL_NAME=all_rand500DCE_1000
MODEL_NAME=$1
python3 tools/print_loss.py $MODEL_NAME --finetune --epoch_num 5
python3 tools/print_score.py $MODEL_NAME --finetune --epoch_num 5
python3 tools/print_use_data.py $MODEL_NAME
