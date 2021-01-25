




for dataset in `ls ../results`
do
    echo $dataset
    Dataname=`echo $dataset | tr '[:lower:]' '[:upper:]'`
    if [ $Dataname == 'COLA' ]; then
        Dataname='CoLA'
    fi
    if [ $Dataname == 'SST2' ]; then
        Dataname='SST-2'
    fi
    if [ $Dataname == 'STSB' ]; then
        Dataname='STS-B'
    fi
    if [ $Dataname == 'MNLI' ]; then
        Dataname='MNLI-m'
    fi
    echo "filename= $Dataname"
    echo "cp ../results/$dataset/test_results_$dataset.txt ./$Dataname.tsv"
    cp ../results/$dataset/test_results_$dataset.txt ./$Dataname.tsv
    python3 ../result_proc.py ./$Dataname.tsv
done

cp ../results/mnli/test_results_mnli-mm.txt ./MNLI-mm.tsv
python3 ../result_proc.py ./MNLI-mm.tsv

echo -e "index\tprediction" > ./AX.tsv
for i in {0..1103};
do
    echo -e "$i\tentailment" >> ./AX.tsv
done

zip -r submission.zip *.tsv
#rm -f *.tsv


