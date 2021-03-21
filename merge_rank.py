import numpy as np
from sklearn.preprocessing import normalize
import json
import scipy.stats
import os

def norm_arr(arr):
    norm = np.linalg.norm(arr)
    normal_arr = arr/norm
    return normal_arr

rank_dir = "./rank_files/"
rank_A = os.path.join(rank_dir, "bert-base")
rank_B = os.path.join(rank_dir, "gpt2-base")
rank_merge = os.path.join(rank_dir, "merge")
ALL_TASK_NAMES = ["mnli", "rte", "qqp", "qnli", "mrpc", "sst2", "cola", "stsb"]
metrics = ['eval_loss', 'entropy']
for task in ALL_TASK_NAMES:
    data_A = json.load(open(os.path.join(rank_A, task+"_rank.json"),"r"))
    data_B = json.load(open(os.path.join(rank_B, task+"_rank.json"),"r"))
    merge_file = os.path.join(rank_merge, task+"_rank.json")
    merge_data = {}
    for metric in metrics:
        d_A = norm_arr(data_A[metric])
        d_B = norm_arr(data_B[metric])
        print(d_A)
        print(d_B)
        merge_data[metric] = [x+y for x,y in zip(d_A,d_B)] 
        merge_data[metric+"_rank"] = list(scipy.stats.rankdata(merge_data[metric]))
    with open(merge_file,"w") as F:
        json.dump(merge_data, F)




