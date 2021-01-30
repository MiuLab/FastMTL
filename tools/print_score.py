import sys
import os
import argparse
import numpy as np
import json

ALL_TASK_METRICS = {"mnli":["accuracy","mm_accuracy"], 
                    "rte":["accuracy"], 
                    "qqp":["accuracy","f1"], 
                    "qnli":["accuracy"], 
                    "mrpc":["accuracy","f1"], 
                    "sst2":["accuracy"], 
                    "cola":["matthews_correlation"],
                    "stsb":["pearson","spearmanr"]}
def score2str(all_score_list, epoch_idx = None):
    all_score_str=""
    score_list = []
    for idx, m_score in enumerate(all_score_list):
        score = ["%.3f" % s for s in m_score]
        score_str = "/".join(score)
        if epoch_idx is not None:
            score_str += " ({})".format(epoch_idx[idx])
        score_list.append(score_str)
    all_score_str = ",".join(score_list)
    return all_score_str

def get_score_from_dir(model_dir):
    #For cal and compare
    all_scores_cal =[]
    for task, metrics in ALL_TASK_METRICS.items():
        file_name = os.path.join(model_dir,"eval_results_{}.txt")
        with open(file_name.format(task),"r") as F:
            data = F.readlines()
        scores_cal = []
        for metric in metrics:
            if metric == "mm_accuracy":
                with open(file_name.format(task+"-mm"),"r") as F:
                    data = F.readlines()
                metric = "accuracy"
            for line in data:
                if ("eval_"+metric) in line:
                    scores_cal.append(100*(float(line.split("=")[1].strip())))
        all_scores_cal.append(scores_cal)

    #return score in float format
    return all_scores_cal
def copy_txt(from_file, to_file):
    with open(from_file,"r") as F:
        data = F.readlines()
    with open(to_file,"w") as F:
        F.writelines(data)

def mv_test_to_best_dir(args, result_dir, best_dir, best_epoch):
    for idx, (task, metrics) in enumerate(ALL_TASK_METRICS.items()):
        epoch_dir = os.path.join(result_dir, "FINETUNE_{}_E{}".format(args.model_name, best_epoch[idx]))
        from_file = os.path.join(epoch_dir,"test_results_{}.txt").format(task)
        to_file = os.path.join(best_dir,"test_results_{}.txt").format(task)
        copy_txt(from_file, to_file)
        if task == "mnli":
            from_file = os.path.join(epoch_dir,"test_results_{}.txt").format(task + "-mm")
            to_file = os.path.join(best_dir,"test_results_{}.txt").format(task + "-mm")
            copy_txt(from_file, to_file)

def main():
    #Parse args
    parser = argparse.ArgumentParser(description='Input model_name, and if Finetune')
    parser.add_argument('model_name', type=str,
                                help='The name of the MT-DNN model')
    parser.add_argument('--finetune', action='store_true',
                                help='If get the best finetune score from all epochs, and write to the best_dir')
    parser.add_argument('--epoch_num', type=int,
                                help='When the --finetune is set, the script requires the max epoch num.')
    parser.add_argument('--best_dir', action='store_true',
                                help='if set, create a dir to put all best files')
    args = parser.parse_args()
    #results are store in './results/'
    result_dir ="./results/"
    mtdnn_dir = os.path.join(result_dir, args.model_name)
    #Check parse args
    #Get MTDNN scores
    mtdnn_score_arr = get_score_from_dir(mtdnn_dir)
    mtdnn_score_str = score2str(mtdnn_score_arr)
    print("[MTDNN] => ", mtdnn_score_str)
    #For Finetune
    if args.finetune:
        print("Get finetune best scores....")
        assert args.epoch_num is not None, "When finetune is set, the --epoch_num needs to be set"
        #Store the scores of every epochs
        all_scores = []
        for e in range(1, 1+args.epoch_num):
            epoch_dir = os.path.join(result_dir, "FINETUNE_{}_E{}".format(args.model_name, e))
            assert os.path.isdir(epoch_dir), "Dir {} not exist error....".format(epoch_dir)
            #Get best FINETUNE scores
            score_arr_e = get_score_from_dir(epoch_dir)
            all_scores.append(score_arr_e)
        # best_scores.shape = (metric_num)
        best_scores = {"best_index":[], "best_epoch":[], "best_score":[], "best_score_str":"", "all_scores":all_scores}
        #Get all metrics best epoch
        for m in range(len(ALL_TASK_METRICS)):
            cmp_scores = [np.prod(score[m]) for score in all_scores]
            best_index = np.argmax(cmp_scores)
            #Epoch is range from 1 to epoch_num+1, index are range from 0 to epoch_num
            best_epoch = best_index + 1
            best_score = all_scores[best_index][m]
            best_scores["best_index"].append(best_index.tolist())
            best_scores["best_epoch"].append(best_epoch.tolist())
            best_scores["best_score"].append(best_score)
        finetune_score_str = score2str(best_scores['best_score'], best_scores['best_epoch'])
        best_scores['best_score_str'] = finetune_score_str
        print("[FINETUNE] => ", finetune_score_str)
        best_dir = os.path.join(result_dir, "FINETUNE_{}_BEST".format(args.model_name))
        #Make best dir
        if args.best_dir:
            if not os.path.isdir(best_dir):
                os.mkdir( best_dir);
            mv_test_to_best_dir(args, result_dir, best_dir, best_scores["best_epoch"])
            #JSON file for best scores
            best_scores_json = os.path.join(best_dir, "best_scores.json")
            with open(best_scores_json,"w") as F:
                json.dump(best_scores, F)
        



        
        






if __name__ == '__main__':
    main()













