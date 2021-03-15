SEED=("1" "2" "3")
NUM=("1000" "2000")

for N in "${NUM[@]}"
do
    for S in "${SEED[@]}"
    do
        ./RAND${S}.sh $N
    done
done

