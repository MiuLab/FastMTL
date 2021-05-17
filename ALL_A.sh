TYPE=("DBCE" "DCE")
SEED=("4")
#NUM=("500" "1000" "2000" "4000" "8000" "16000" "32000" "64000" "128000" "256000" "512000" "600000" "700000" "800000")
NUM=("500")

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

