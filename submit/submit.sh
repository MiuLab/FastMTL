


MODEL_NAME="all"
POSTFIX=0
FILEDIR="../results/${MODEL_NAME}_${POSTFIX}"
mkdir $FILEDIR/submission 
cp WNLI.tsv $FILEDIR/submission/

for dataset in `cat ./datasets`
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
    echo "cp $FILEDIR/test_results_$dataset.txt $FILEDIR/submission/$Dataname.tsv"

    cp $FILEDIR/test_results_$dataset.txt $FILEDIR/submission/$Dataname.tsv
    python3 ../result_proc.py $FILEDIR/submission/$Dataname.tsv
done

cp $FILEDIR/test_results_mnli-mm.txt $FILEDIR/submission/MNLI-mm.tsv

echo -e "index\tprediction" > $FILEDIR/submission/AX.tsv
for i in {0..1103};
do
    echo -e "$i\tentailment" >> $FILEDIR/submission/AX.tsv
done
cd $FILEDIR/submission/
zip -r -j ../submission.zip *.tsv
#rm -f *.tsv


