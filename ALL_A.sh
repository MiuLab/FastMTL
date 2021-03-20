TYPE=("DBCE" "DCE")
SEED=("0" "1" "2")
NUM=("1000" "2000")

for T in "${TYPE[@]}"
do
    for N in "${NUM[@]}"
    do
        for S in "${SEED[@]}"
        do
            ./BCE_CE.sh $T $S $N
        done
    done
done

