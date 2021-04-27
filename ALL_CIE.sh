# Before running, need to mv run_glue_CIE.py to run_glue.py
TYPE=("DCE")
SEED=("2")
NUM=("100000")
SEED_BASE=50

for T in "${TYPE[@]}"
do
    for N in "${NUM[@]}"
    do
        for S in "${SEED[@]}"
        do
            ./BCE_CE_CIE.sh $T $S $N $SEED_BASE
        done
    done
done

