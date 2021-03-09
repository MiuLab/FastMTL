import random
import sys
import json
import os

subdataset_num = int(sys.argv[1])
seed = int(sys.argv[2])
random.seed(seed)
subdataset_file = "subdataset_dir/{}_{}.json".format(subdataset_num,seed)
ALL_TASK={'mnli' : 392702,
        'rte' : 2490,
        'qqp' : 363849,
        'qnli' : 104743,
        'mrpc' : 3668,
        'sst2' : 67349,
        'cola' : 8551,
        'stsb' : 5749}
ALL_LIST={}
for task_name, task_num in ALL_TASK.items():
    task_use_list = random.sample(range(task_num), subdataset_num)
    ALL_LIST[task_name] = task_use_list

with open(subdataset_file,"w") as F:
    json.dump(ALL_LIST, F)

    
