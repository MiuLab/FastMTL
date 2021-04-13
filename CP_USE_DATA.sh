TASK=("rte" "mrpc" "stsb")
TYPE=("_rand" "DBCE" "DCE")
SEED=("0" "1" "2")
NUM=("20000")

for task in "${TASK[@]}"
do
    for T in "${TYPE[@]}"
    do
        T_NAME="$T"
        if [ "$T" = "_rand" ]; then
            T_NAME="RANDOM"
        fi

        for N in "${NUM[@]}"
        do
            for S in "${SEED[@]}"
            do
                S_NAME="$S"
                if [ "$S" = "0" ]; then
                    S_NAME=""
                fi
                cp -r results/all_rand500${T}${S_NAME}_${task}${N}/use_data.txt ../USE_DATA_NUM/${N}/${task}_${T_NAME}_${S}.txt
            done
        done
    done
done

