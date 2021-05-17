TYPE=("DBCE" "DCE")
SEED=("4")
SEED_BASE=200

for T in "${TYPE[@]}"
do
    for S in "${SEED[@]}"
    do
        ./train_TD.sh $T $S $SEED_BASE
    done
done

