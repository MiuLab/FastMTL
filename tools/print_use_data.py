import sys
import os
model_name = sys.argv[1]
ALL_TASK_NAMES=["rte", "mrpc", "stsb"]
print("Print Use data, copy to excel...")
for task in ALL_TASK_NAMES:
    print_str=["{}-{}".format(model_name, task)]

    with open(os.path.join("results/"+model_name,"use_data_{}.txt".format(task)),"r") as F:
        lines = F.readlines()
        del lines[0]
        for line in lines:
            print_str.append(line.split(" ")[-1].replace("\n","").strip())
    print_str = ",".join(print_str)
    print(print_str)




