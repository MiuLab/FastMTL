import numpy as np
from sklearn.preprocessing import normalize
import json
import scipy.stats
import random
import os
import sys

model_dir = sys.argv[1]
task_disc_file = os.path.join(model_dir, "task_disc_pred.json")
rank_dir = os.path.join(model_dir, "task_disc_rank")
if not os.path.exists(rank_dir):
    os.mkdir( rank_dir, 0o777 );
    
ALL_TASK_NAMES = ["mnli", "rte", "qqp", "qnli", "mrpc", "sst2", "cola", "stsb"]
ALL_TASK_LEN={
    'mnli' : 392702,
    'rte' : 2490,
    'qqp' : 363849,
    'qnli' : 104743,
    'mrpc' : 3668,
    'sst2' : 67349,
    'cola' : 8551,
    'stsb' : 5749}
#Load Data first
all_data = {}
#init all data, each item will be dump to a file
for task in ALL_TASK_NAMES:
    all_data[task]={}
task_disc_data = json.load(open(task_disc_file,"r"))
for TYPE in ["Normal", "Random"]:
    for task_idx, task in enumerate(ALL_TASK_NAMES):
        print("-------------running {}-------------".format(task))
        all_other_data_val = []
        for other_task in ALL_TASK_NAMES:
            if other_task == task: #do other tasks
                continue
            all_other_data_val += [ d[task_idx] for d in task_disc_data[other_task]['predictions'] ]
        #rank all together, all other tasks data share same ranks
        #rank reverse, small value get high rank
        #For random, init unique random numbers
        if TYPE == "Random":
            all_other_data_val = random.sample(range(len(all_other_data_val)), len(all_other_data_val))
        all_other_data_rank = list(scipy.stats.rankdata([ -i for i in all_other_data_val]))
        cnt_start = 0
        for other_task in ALL_TASK_NAMES:
            if other_task == task: #do other tasks
                continue
            print("Process [{}] for [{}]".format(other_task,task))
            other_data_unused = task_disc_data[other_task]['unused_list']
            unused_len = len(other_data_unused)
            other_data_rank = all_other_data_rank[cnt_start:cnt_start+unused_len]
            cnt_start += unused_len
            rank_data = [-1]*ALL_TASK_LEN[other_task]
            for unused_idx, unused_rank in zip(other_data_unused, other_data_rank):
                rank_data[unused_idx] = unused_rank
            if TYPE == "Random":
                all_data[other_task]["for_"+task+"_random_rank"] = rank_data
            else:
                all_data[other_task]["for_"+task+"_rank"] = rank_data

for task in ALL_TASK_NAMES:
    task_rank_file = os.path.join(rank_dir, task+"_rank.json")
    with open(task_rank_file,"w") as F:
        json.dump(all_data[task], F)


#    merge_file = os.path.join(rank_merge, task+"_rank.json")
#    merge_data = {}
#    for metric in metrics:
#        d_A = norm_arr(data_A[metric])
#        d_B = norm_arr(data_B[metric])
#        print(d_A)
#        print(d_B)
#        merge_data[metric] = [x+y for x,y in zip(d_A,d_B)] 
#        merge_data[metric+"_rank"] = list(scipy.stats.rankdata(merge_data[metric]))
#    with open(merge_file,"w") as F:
#        json.dump(merge_data, F)




