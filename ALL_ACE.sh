SEED=("1" "2" "3")
NUM=("10000" "20000")

for N in "${NUM[@]}"
do
    for S in "${SEED[@]}"
    do
        ./CE${S}.sh $N
    done
done

