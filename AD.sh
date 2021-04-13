TYPE=("rand")
SEED=("0" "1" "2")
NUM=("100000" "200000")


for T in "${TYPE[@]}"
do
    for N in "${NUM[@]}"
    do
        for S in "${SEED[@]}"
        do
            ./GET_DEV_SCORE.sh $T $S $N
        done
    done
done
