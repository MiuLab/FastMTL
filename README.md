# Reproduce
## To Reproduce our experiment results in  Fig 3,4,5
### Environment Settings
1. Make sure the python version >= 3.7
2. pip3 install -r requirements.txt
3. The default CUDA is set to 0. If you want to change the used CUDA, modify the CUDA Variable in BCE_CE.sh, train_TD.sh, RAND.sh
### Running Experiment
The default experiment settings:
    ./ALL_TD.sh
        Train a BCE-TD-MTDNN model and CE-TD-MTDNN model. Notice that this script should be run before running ALL_R.sh and ALL_A.sh
    ./ALL_R.sh
        Train a Random-Sampling with TO-MTDNN using total 500 auxiliary data. To change the data amount, modify $NUM$ in ALL_R.sh
    ./ALL_A.sh
        Train two TO-MTDNN with TO-MTDNN using total 500 auxiliary data sampled from BCE-TD-MTDNN and CE-TD-MTDNN. To change the data amount, modify $NUM$ in ALL_A.sh
    ./Train_ALL.sh
        Execute all the above scripts.
### Output
    When the experiment is done, the outputs are store in the following folders:
        Models --> results/
        TD-MTDNN T-SNE visualization --> figures/
        The final submission file (For GLUE Benchmark) --> final_submission/
        !! Notice that in the final submission scores, only the RTE, MRPC and STS-B scores are real. Since the submission requires all tasks data, we copy the files for other tasks in FAKE_TEST3/ to make the submission zip.

