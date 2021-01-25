import sys
import os
model_dir = sys.argv[1]
print(model_dir)

ALL_TASK_METRICS = {"mnli":["accuracy","mm_accuracy"], 
        "rte":["accuracy"], 
        "qqp":["accuracy","f1"], 
        "qnli":["accuracy"], 
        "mrpc":["accuracy","f1"], 
        "sst2":["accuracy"], 
        "cola":["matthews_correlation"],
        "stsb":["pearson","spearmanr"]}

all_scores = []
for task, metrics in ALL_TASK_METRICS.items():
    file_name = os.path.join(model_dir,"/eval_results_{}.txt")
    print(file_name)
    with open(file_name.format(task),"r") as F:
        data = F.readlines()
    scores = []
    for metric in metrics:
        if metric == "mm_accuracy":
            with open(file_name.format(task+"-mm"),"r") as F:
                data = F.readlines()
            metric = "accuracy"
        for line in data:
            if ("eval_"+metric) in line:
                scores.append("%.3f" % (100*(float(line.split("=")[1].strip()))))
    metric_str = " / ".join(scores)
    all_scores.append(metric_str)

all_str = ",".join(all_scores)
print(all_str)





