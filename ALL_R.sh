SEED=("0" "1" "2")
NUM=("1000" "2000")

for N in "${NUM[@]}"
do
    for S in "${SEED[@]}"
    do
        ./RAND.sh $S $N
    done
done

