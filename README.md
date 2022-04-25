# Efficient Multi-Task Auxiliary Learning: Selecting Auxiliary Data by Feature Similarity

This repository contains the code for reproducing [our paper](https://www.csie.ntu.edu.tw/~yvchen/doc/EMNLP21_FastMTL.pdf).

## Overview

Multi-task auxiliary learning utilizes auxiliary tasks to improve a primary task, while (1) selecting beneficial auxiliary tasks for a primary task is nontrivial, and (2) when the auxiliary datasets are large, training on all data becomes time-expensive and impractical. Therefore, we propose a time-efficient sampling method to select the data most beneficial to the primary task. The experiments on GLUE show 12x speed improvement compared to fully-trained MT-DNN. The following figure is an illustration of our framework.
![image](https://user-images.githubusercontent.com/2268109/165054576-0205ac32-5515-4e1a-be37-73d7ea7d3e1d.png)

## To Reproduce our experiment results in Fig 3-5
### Environment Settings
1. Make sure the python version >= 3.7
2. pip3 install -r requirements.txt
3. The default CUDA is set to 0. If you want to change the used CUDA, modify the CUDA Variable in BCE_CE.sh, train_TD.sh, RAND.sh
### Running Experiment
The default experiment settings:
* Train a BCE-TD-MTDNN model and CE-TD-MTDNN model. Notice that this script should be run before running ALL_R.sh and ALL_A.sh
```bash
./ALL_TD.sh
```
* Train a Random-Sampling with TO-MTDNN using total 500 auxiliary data. To change the data amount, modify $NUM$ in ALL_R.sh
```bash
./ALL_R.sh
```
* Train two TO-MTDNN with TO-MTDNN using total 500 auxiliary data sampled from BCE-TD-MTDNN and CE-TD-MTDNN. To change the data amount, modify $NUM$ in ALL_A.sh
```bash
./ALL_A.sh
```
* Execute all the above scripts.
```bash
./Train_ALL.sh
```

### Output
    When the experiment is done, the outputs are store in the following folders:
        Models --> results/
        TD-MTDNN T-SNE visualization --> figures/
        The final submission file (For GLUE Benchmark) --> final_submission/
        !! Notice that in the final submission scores, only the RTE, MRPC and STS-B scores are real. Since the submission requires all tasks data, we copy the files for other tasks in FAKE_TEST3/ to make the submission zip.


## Citation

Please cite our paper if you use SimCSE in your work:

```bibtex
@inproceedings{kung2021efficient,
   title={Efficient Multi-Task Auxiliary Learning: Selecting Auxiliary Data by Feature Similarity},
   author={Po-Nien Kung, Sheng-Siang Yin, Yi-Cheng Chen, Tse-Hsuan Yang, and Yun-Nung Chen},
   booktitle={Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
   year={2021}
}
```
