# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Finetuning the library models for sequence classification on GLUE."""
# You can also adapt this script on your own text classification task. Pointers for this are left as comments.

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from datasets import load_dataset, load_metric

import transformers
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EvalPrediction,
    HfArgumentParser,
    PretrainedConfig,
    #Modify --> use custom trainer
    #Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from transformers.trainer_utils import is_main_process
# MODIFY --> Import new BertForSequenceClassification
from modeling_bert import BertForSequenceClassification
# MODIFY --> Import custom dataset to pach train dataset
from custom_dataset import *
#MODIFY --> use new trianer
from custom_trainer import Trainer
#MODIFY --> import json, numpy
import numpy as np
import json
#MODIFY --> import bert hiddenstate visualization
from vis_hidden import *


#MODIFY --> change order, following the task id order
task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "rte": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "qnli": ("question", "sentence"),
    "mrpc": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "cola": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

logger = logging.getLogger(__name__)


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the task to train on: " + ", ".join(task_to_keys.keys())},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(default=None, metadata={"help": "A csv or a json file containing the test data."})
    #MODIFY --> add argument for using percent of data
    use_data_percent: int = field(
        default=100,
        metadata={
            "help": "The percent of data to use, if no load_rank_dir is specified, use random"
        },
    )
    use_data_abs: int = field(
        default=-1,
        metadata={
            "help": "Use absolute data num"
        },
    )
    load_rank_dir: Optional[str] = field(
        default=None,
        metadata={"help": "The dir includes rank files"}
    )
    rank_type: Optional[str] = field(
        default=None,
        metadata={"help": "Type of ranking, the key of the rank file"}
    )
    vis_hidden: Optional[bool] = field(
        default=False,
        metadata={"help" : "When doing eval, if want to store hidden img"}
    )
    vis_hidden_file: Optional[str] = field(
        default="vis_hidden.png",
        metadata={
            "help" : "When doing eval, if want to store hidden img, specify the store file name"
        }
    )
    vis_hidden_num: Optional[int] = field(
        default=-1,
        metadata={
            "help" : "If train_dataset isn;t 100, use unused, else use eval. To store hidden img, specify the show datanum"
        }
    )
    train_task_disc: Optional[bool] = field(
        default=False,
        metadata={"help" : "Only specify this in training mode, if specified, train task_discriminator"}
    )
    train_disc_CE: Optional[bool] = field(
        default=False,
        metadata={"help" : "If use CE to train task disc, default use BCE"}
    )
    task_disc_num: Optional[int] = field(
        default=8,
        metadata={"help" : "Number of task to disc, default is 8(GLUE tasks num)"}
    )
    do_predict_task: Optional[bool] = field(
        default=False,
        metadata={"help" : "If train_dataset isn;t 100, use unused, else use eval. If predict task type"}
    )
    mtdnn_target_task: Optional[str] = field(
        default="None",
        metadata={
            "help" : "if want to train the mtdnn with a target task"
        }
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            #MODIFY --> enable task_name == all
            #if self.task_name not in task_to_keys.keys():
            if self.task_name not in task_to_keys.keys() and self.task_name != "all":
                raise ValueError("Unknown task, you should pick one in " + ",".join(task_to_keys.keys()))
        elif self.train_file is None or self.validation_file is None:
            raise ValueError("Need either a GLUE task or a training/validation file.")
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    #MODIFY --> get real train_batch_size
    TrainingArguments.real_train_batch_size = TrainingArguments.train_batch_size
    TrainingArguments.train_batch_size = 1
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. "
            "Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if is_main_process(training_args.local_rank) else logging.WARN,
    )

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or specify a GLUE benchmark task (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use as labels the column called 'label' and as pair of sentences the
    # sentences in columns called 'sentence1' and 'sentence2' if such column exists or the first two columns not named
    # label if at least two columns are provided.
    #
    # If the CSVs/JSONs contain only one non-label column, the script does single sentence classification on this
    # single column. You can easily tweak this behavior (see below)
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.

    #MODIFY --> CHECK use_data_percent and use_data_abs not specify at the same time
    print(data_args.use_data_percent, data_args.use_data_abs)
    assert not((data_args.use_data_percent < 100) and (data_args.use_data_abs>0)), "Error, --use_data_percent and --use_data_abs are exclusive args"
    # MODIFY --> get list of dataset, add datasets_dict, num_labels_list, label_list_dict
    # !!! data_args.task_name == "all" 時不適用自己load的dataset(data_args.train_file等)
    datasets_dict = {}
    num_labels_dict = {}
    label_list_dict = {}
    is_regression_dict = {}
    if data_args.task_name == "all":
        ALL_TASK_NAMES = ["mnli", "rte", "qqp", "qnli", "mrpc", "sst2", "cola", "stsb"]
    else:
        ALL_TASK_NAMES = [data_args.task_name]
    for task_name in ALL_TASK_NAMES:
        data_args.task_name = task_name
        if data_args.task_name is not None:
            # Downloading and loading a dataset from the hub.
            datasets = load_dataset("glue", data_args.task_name)
        else:
            # Loading a dataset from your local files.
            # CSV/JSON training and evaluation files are needed.
            data_files = {"train": data_args.train_file, "validation": data_args.validation_file}

            # Get the test dataset: you can provide your own CSV/JSON test file (see below)
            # when you use `do_predict` without specifying a GLUE benchmark task.
            if training_args.do_predict:
                if data_args.test_file is not None:
                    train_extension = data_args.train_file.split(".")[-1]
                    test_extension = data_args.test_file.split(".")[-1]
                    assert (
                        test_extension == train_extension
                    ), "`test_file` should have the same extension (csv or json) as `train_file`."
                    data_files["test"] = data_args.test_file
                else:
                    raise ValueError("Need either a GLUE task or a test file for `do_predict`.")

            for key in data_files.keys():
                logger.info(f"load a local file for {key}: {data_files[key]}")

            if data_args.train_file.endswith(".csv"):
                # Loading a dataset from local csv files
                datasets = load_dataset("csv", data_files=data_files)
            else:
                # Loading a dataset from local json files
                datasets = load_dataset("json", data_files=data_files)
        # See more about loading any type of standard or custom dataset at
        # https://huggingface.co/docs/datasets/loading_datasets.html.

        # Labels
        label_list = None
        if data_args.task_name is not None:
            is_regression = data_args.task_name == "stsb"
            if not is_regression:
                label_list = datasets["train"].features["label"].names
                num_labels = len(label_list)
            else:
                num_labels = 1
        else:
            # Trying to have good defaults here, don't hesitate to tweak to your needs.
            is_regression = datasets["train"].features["label"].dtype in ["float32", "float64"]
            if is_regression:
                num_labels = 1
            else:
                # A useful fast method:
                # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.unique
                label_list = datasets["train"].unique("label")
                label_list.sort()  # Let's sort it for determinism
                num_labels = len(label_list)
        # MODIFY --> add datasets, label_list and num_labels to lists
        datasets_dict[data_args.task_name] = datasets
        num_labels_dict[data_args.task_name] = num_labels
        label_list_dict[data_args.task_name] = label_list
        is_regression_dict[data_args.task_name] = is_regression

    # Load pretrained model and tokenizer
    #
    # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        #MODIFY --> don't set num labels
        #num_labels=num_labels,
        #finetuning_task=data_args.task_name,
        finetuning_task="all",
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    #MODIFY --> add num labels list and all task names
    config.num_labels_dict = num_labels_dict
    config.all_task_names = ALL_TASK_NAMES
    config.train_task_disc = data_args.train_task_disc
    config.task_disc_num = data_args.task_disc_num

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # MODIFY, change to our defined Bert model structure
    #model = AutoModelForSequenceClassification.from_pretrained(
    model = BertForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    #Update some args, such as vis_hidden=None and train_task_disc
    model.update_args(data_args)

    # MODIFY --> make train_dataset_list and test_dataset_list
    train_dataset_list = []
    unused_idx_list = []
    unused_train_dataset_list = []
    eval_dataset_list = [] 
    test_dataset_list = [] 
    task_use_dict = {} 
    all_use_list_dict = {}
    #Handle exception for do_predict_task_disc
    EXCEPTION_DICT={}
    #when data_args.mtdnn_target_task is set
    ignore_list = []
    for task_name in ALL_TASK_NAMES:
        # ----------------------------------------- set param start -----------------------------------------
        data_args.task_name = task_name
        label_list = label_list_dict[data_args.task_name]
        datasets = datasets_dict[data_args.task_name]
        num_labels = num_labels_dict[data_args.task_name]
        is_regression = is_regression_dict[data_args.task_name]
        # ----------------------------------------- set param end -----------------------------------------
        # Preprocessing the datasets
        if data_args.task_name is not None:
            sentence1_key, sentence2_key = task_to_keys[data_args.task_name]
        else:
            # Again, we try to have some nice defaults but don't hesitate to tweak to your use case.
            non_label_column_names = [name for name in datasets["train"].column_names if name != "label"]
            if "sentence1" in non_label_column_names and "sentence2" in non_label_column_names:
                sentence1_key, sentence2_key = "sentence1", "sentence2"
            else:
                if len(non_label_column_names) >= 2:
                    sentence1_key, sentence2_key = non_label_column_names[:2]
                else:
                    sentence1_key, sentence2_key = non_label_column_names[0], None

        # Padding strategy
        if data_args.pad_to_max_length:
            padding = "max_length"
        else:
            # We will pad later, dynamically at batch creation, to the max sequence length in each batch
            padding = False

        # MODIFY --> Ignore this case
        model.config.label2id = None
        # Some models have set the order of the labels to use, so let's make sure we do use it.
        label_to_id = None
        #if (
        #    model.config.label2id != PretrainedConfig(num_labels=num_labels).label2id
        #    and data_args.task_name is not None
        #    and is_regression
        #):
        #    # Some have all caps in their config, some don't.
        #    label_name_to_id = {k.lower(): v for k, v in model.config.label2id.items()}
        #    if list(sorted(label_name_to_id.keys())) == list(sorted(label_list)):
        #        label_to_id = {i: label_name_to_id[label_list[i]] for i in range(num_labels)}
        #    else:
        #        logger.warn(
        #            "Your model seems to have been trained with labels, but they don't match the dataset: ",
        #            f"model labels: {list(sorted(label_name_to_id.keys()))}, dataset labels: {list(sorted(label_list))}."
        #            "\nIgnoring the model labels as a result.",
        #        )
        if data_args.task_name is None and not is_regression:
            label_to_id = {v: i for i, v in enumerate(label_list)}

        def preprocess_function(examples):
            # Tokenize the texts
            args = (
                (examples[sentence1_key],) if sentence2_key is None else (examples[sentence1_key], examples[sentence2_key])
            )
            result = tokenizer(*args, padding=padding, max_length=data_args.max_seq_length, truncation=True)

            # Map labels to IDs (not necessary for GLUE tasks)
            if label_to_id is not None and "label" in examples:
                result["label"] = [label_to_id[l] for l in examples["label"]]
            return result
        #MODIFY --> use cache
        datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=not data_args.overwrite_cache)
        #datasets = datasets.map(preprocess_function, batched=True, load_from_cache_file=False)
        # MODIFY --> Re-assign datasets after mapping
        datasets_dict[data_args.task_name] = datasets

        train_dataset = datasets["train"]
        eval_dataset = datasets["validation_matched" if data_args.task_name == "mnli" else "validation"]
        #MODIFY--> Init unused dataset ausing eval dataset
        unused_train_dataset = eval_dataset
        unused_list = [-1]
        #MODIFY --> add use percent of data
        if (data_args.use_data_percent < 100) or (data_args.use_data_abs > 0):
            #The setting will match the setting in shell script, use two divide with floor...
            if data_args.use_data_percent < 100:
                use_percent_div = 100//data_args.use_data_percent
                use_datanum = len(train_dataset)//use_percent_div
            if data_args.use_data_abs > 0:
                use_datanum = data_args.use_data_abs
            #use_list
            if (data_args.load_rank_dir is None) or (data_args.load_rank_dir=="None"): #Use random
                use_list = random.sample(range(0, len(train_dataset)), use_datanum)
            else: #use rank
                rank_file = os.path.join(data_args.load_rank_dir, "{}_rank.json".format(data_args.task_name)) 
                with open(rank_file, "r") as F:
                    ranking = json.load(F)
                if data_args.mtdnn_target_task !="None": #use target task rank
                    if task_name == data_args.mtdnn_target_task:#the target task    
                        #use all
                        use_list = [i for i in range(len(train_dataset))]
                    else: #other task
                        print(ranking.keys())
                        if data_args.rank_type == "random":
                            use_rank = ranking["for_"+data_args.mtdnn_target_task+"_random_rank"]
                        else:
                            use_rank = ranking["for_"+data_args.mtdnn_target_task+"_rank"]
                        # the rank here is revers, we use the smallest rank
                        # we use the use_data_abs, the rank start from 1, so we use <=
                        use_list = []
                        for idx, rank in enumerate(use_rank):
                            if rank > 0 and rank <= data_args.use_data_abs: # The used(trained) rank will be -1 
                                use_list.append(idx)
                else:
                    #From small to big, forr bert, gpt2, merge ranks
                    use_rank = np.array(ranking[data_args.rank_type])
                    use_rank_argmax = np.argsort(use_rank) 
                    use_list = use_rank_argmax[len(train_dataset)-use_datanum:].tolist()
                print("use rank(first 10): ", use_list[:10])
                print("rank list(first 10, should be accending): ", [use_rank[i] for i in use_list[:10]])
            #CHECK use_list unique
            assert len(use_list) == len(set(use_list)), "ERROR, use_list is not unique...."
            #STORE USE LIST
            all_use_list_dict[data_args.task_name] = use_list
            unused_list = list(set([i for i in range(len(train_dataset))])-set(use_list))
            #Handle prediction error in trainer, when last batch size has only 1 element...
            if len(unused_list) % training_args.per_device_eval_batch_size == 1:#
                unused_list.append(use_list[0]) #add a pad element, remove afterr prediction end
                EXCEPTION_DICT[data_args.task_name] = 1
            if len(unused_list) ==0:#handel exception, have at least 1 data
                unused_list = [0]
            if len(use_list) ==0:#handel exception, have at least 1 data
                use_list = [0]
                ignore_list.append(data_args.task_name)
            unused_train_dataset = train_dataset.select(unused_list)
            train_dataset = train_dataset.select(use_list)




        if data_args.task_name is not None or data_args.test_file is not None:
            test_dataset = datasets["test_matched" if data_args.task_name == "mnli" else "test"]

        # Log a few random samples from the training set:
        #for index in random.sample(range(len(train_dataset)), 3):
        #    logger.info(f"Sample {index} of the training set: {train_dataset[index]}.")
        
        #MODIFY --> append datasets
        train_dataset_list.append(train_dataset)
        unused_train_dataset_list.append(unused_train_dataset)
        unused_idx_list.append(unused_list)
        eval_dataset_list.append(eval_dataset)
        test_dataset_list.append(test_dataset)

    # 這邊沒有用成回圈的形式，eval 的部分改在eval函式那邊，用迴圈跑每一種task
    # MODIFY --> 這邊get metrics 寫在Eval 和 Predict裡面
    # Get the metric function
    #if data_args.task_name is not None:
    #    metric = load_metric("glue", data_args.task_name)
    # TODO: When datasets metrics include regular accuracy, make an else here and remove special branch from
    # compute_metrics

    # You can define your custom compute_metrics function. It takes an `EvalPrediction` object (a namedtuple with a
    # predictions and label_ids field) and has to return a dictionary string to float.
    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)
        if data_args.task_name is not None:
            result = metric.compute(predictions=preds, references=p.label_ids)
            if len(result) > 1:
                result["combined_score"] = np.mean(list(result.values())).item()
            return result
        elif is_regression:
            return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
        else:
            return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

    # Initialize our Trainer
    #MODIFY --> use new dataset to pack train datasets, tokenizer is for the datacollator
    train_dataset_all = MergeDataset(training_args, train_dataset_list, ALL_TASK_NAMES)
    #if save each epoch, training_args.save_steps = -1
    if training_args.save_steps == -1:
        training_args.save_steps = len(train_dataset_all)
    #MODIFY --> use simple data collator in trainer when training.
    custom_data_collator = CustomDataCollator()
    #MODIFY --> For task disc, get pos weight
    model.task_disc_pos_weight = train_dataset_all.get_task_disc_pos_weight()
    model.train_disc_CE = data_args.train_disc_CE
    #MODIFY--> set ignore
    if data_args.mtdnn_target_task != "None":
        #ignore those empty task
        model.ignore_list = ignore_list
        print(ignore_list)
        for task_name, train_dataset in zip(ALL_TASK_NAMES, train_dataset_list):
            dat_num = len(train_dataset)
            if task_name in ignore_list:
                dat_num = 0
            print("[{}] -- used data: {}".format(task_name, dat_num))

    trainer = Trainer(
        model=model,
        args=training_args,
        #MODIFY --> train_dataset --> train_dataset_list, 需要改training process
        train_dataset=train_dataset_all,
        #MODIFY --> 移除eval的部分，在Eval函式那邊再給eval_dataset 和 compute_metrics
        #eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it if we already did the padding.
        data_collator=default_data_collator if data_args.pad_to_max_length else None,
    )
    trainer.train_data_collator = custom_data_collator

    # Training
    if training_args.do_train:
        train_result = trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        metrics = train_result.metrics

        trainer.save_model()  # Saves the tokenizer too for easy upload

        output_train_file = os.path.join(training_args.output_dir, "train_results.txt")
        if trainer.is_world_process_zero():
            with open(output_train_file, "w") as writer:
                logger.info("***** Train results *****")
                for key, value in sorted(metrics.items()):
                    logger.info(f"  {key} = {value}")
                    writer.write(f"{key} = {value}\n")

            # Need to save the state, since Trainer.save_model saves only the tokenizer with the model
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))

    # MODIFY --> 這邊注意的是，Eval 和predict 都是用原本的方法來跑，不像train會用雙層dataloader
    # MODIFY --> Eval and predict needs "model.task_name to know which task it is running"
    #STORE USE LIST
    output_use_list = os.path.join(training_args.output_dir, "use_list.json")
    print("Store all_use_list_dict to  {} ".format(output_use_list))
    with open(output_use_list,"w") as F:
        json.dump(all_use_list_dict ,F)
    if data_args.mtdnn_target_task != "None":
        output_use_txt = os.path.join(training_args.output_dir, "use_data.txt")
        with open(output_use_txt,"w") as F:
            print(ignore_list,file=F)
            for task_name, train_dataset in zip(ALL_TASK_NAMES, train_dataset_list):
                dat_num = len(train_dataset)
                if task_name in ignore_list:
                    dat_num = 0
                print("[{}] -- used data: {}".format(task_name, dat_num), file=F)

    # Evaluation
    eval_results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        #MODIFY --> loop to eval all tasks in TASK_NAME_LIST
        for task_name, eval_dataset_store in zip(ALL_TASK_NAMES, eval_dataset_list):
            if data_args.mtdnn_target_task != "None":
                if task_name != data_args.mtdnn_target_task and data_args.mtdnn_target_task != "random":
                    continue #skip
            # ----------------------------------------- set param start -----------------------------------------
            data_args.task_name = task_name
            label_list = label_list_dict[data_args.task_name]
            datasets = datasets_dict[data_args.task_name]
            num_labels = num_labels_dict[data_args.task_name]
            is_regression = is_regression_dict[data_args.task_name]
            # ----------------------------------------- set param end -----------------------------------------
            trainer.model.task_name = data_args.task_name
            # MODIFY 在這邊get metrics
            # Get the metric function
            if data_args.task_name is not None:
                metric = load_metric("glue", data_args.task_name)
            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [data_args.task_name]
            eval_datasets = [eval_dataset_store]
            if data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                eval_datasets.append(datasets["validation_mismatched"])

            for eval_dataset, task in zip(eval_datasets, tasks):
                eval_result = trainer.evaluate(eval_dataset=eval_dataset)
                output_eval_file = os.path.join(training_args.output_dir, f"eval_results_{task}.txt")
                if trainer.is_world_process_zero():
                    with open(output_eval_file, "w") as writer:
                        logger.info(f"***** Eval results {task} *****")
                        for key, value in sorted(eval_result.items()):
                            logger.info(f"  {key} = {value}")
                            writer.write(f"{key} = {value}\n")

                eval_results.update(eval_result)


    if training_args.do_predict:
        logger.info("*** Test ***")

        #MODIFY --> loop to eval all tasks in TASK_NAME_LIST
        for task_name, test_dataset_store in zip(ALL_TASK_NAMES, test_dataset_list):
            if data_args.mtdnn_target_task != "None":
                if task_name != data_args.mtdnn_target_task:
                    continue #skip
            # ----------------------------------------- set param start -----------------------------------------
            data_args.task_name = task_name
            label_list = label_list_dict[data_args.task_name]
            datasets = datasets_dict[data_args.task_name]
            num_labels = num_labels_dict[data_args.task_name]
            is_regression = is_regression_dict[data_args.task_name]
            # ----------------------------------------- set param end -----------------------------------------
            trainer.model.task_name = data_args.task_name
            # MODIFY 在這邊get metrics
            # Get the metric function
            if data_args.task_name is not None:
                metric = load_metric("glue", data_args.task_name)
            # Loop to handle MNLI double evaluation (matched, mis-matched)
            tasks = [data_args.task_name]
            test_datasets = [test_dataset_store]
            if data_args.task_name == "mnli":
                tasks.append("mnli-mm")
                test_datasets.append(datasets["test_mismatched"])

            for test_dataset, task in zip(test_datasets, tasks):
                # Removing the `label` columns because it contains -1 and Trainer won't like that.
                test_dataset.remove_columns_("label")
                predictions = trainer.predict(test_dataset=test_dataset).predictions
                predictions = np.squeeze(predictions) if is_regression else np.argmax(predictions, axis=1)

                output_test_file = os.path.join(training_args.output_dir, f"test_results_{task}.txt")
                if trainer.is_world_process_zero():
                    with open(output_test_file, "w") as writer:
                        logger.info(f"***** Test results {task} *****")
                        writer.write("index\tprediction\n")
                        for index, item in enumerate(predictions):
                            if is_regression:
                                writer.write(f"{index}\t{item:3.3f}\n")
                            else:
                                item = label_list[item]
                                writer.write(f"{index}\t{item}\n")
    # MODIFY--> to vis hidden, tell model
    if data_args.vis_hidden:
        trainer.model.vis_hidden=True
        #INIT task hidden state dict
        trainer.model.tasks_hs_dict={}
        for task_name in ALL_TASK_NAMES:
            trainer.model.tasks_hs_dict[task_name]=[]
        all_hidden_states = []
        all_y=[]
        cnt_task_for_vis_hid = 0

    if data_args.do_predict_task:
        trainer.model.do_predict_task=True
    if data_args.do_predict_task or data_args.vis_hidden:
        logger.info("*** Pred Task Disc ***")
        all_task_pred = {}

        #MODIFY --> loop to eval all tasks in TASK_NAME_LIST
        for task_name, test_dataset_store, unused_list in zip(ALL_TASK_NAMES, unused_train_dataset_list, unused_idx_list):

            # ----------------------------------------- set param start -----------------------------------------
            data_args.task_name = task_name
            datasets = datasets_dict[data_args.task_name]
            # ----------------------------------------- set param end -----------------------------------------
            trainer.model.task_name = data_args.task_name
            tasks = [data_args.task_name]
            test_dataset = test_dataset_store
            if test_dataset is None:
                continue
            # Removing the `label` columns because it contains -1 and Trainer won't like that.
            test_dataset.remove_columns_("label")
            predictions = trainer.predict(test_dataset=test_dataset).predictions
            if task_name in EXCEPTION_DICT.keys():
                predictions = predictions[:-1] #remove the last pad element
                print("Remove exception element in Task [{}], real prediction len :{}".format(task_name, len(predictions)))
            all_task_pred[task_name] = {"unused_list": unused_list, "predictions":predictions.tolist()}


            #MODIFY -> VIS HIDDEN
            if data_args.vis_hidden:
                if data_args.vis_hidden_num <= 0:
                    all_hidden_states += trainer.model.tasks_hs_dict[task_name]
                    all_y += [cnt_task_for_vis_hid] * len(trainer.model.tasks_hs_dict[task_name])
                else:
                    all_hidden_states += trainer.model.tasks_hs_dict[task_name][:data_args.vis_hidden_num]
                    all_y += [cnt_task_for_vis_hid] * len(trainer.model.tasks_hs_dict[task_name][:data_args.vis_hidden_num])
                cnt_task_for_vis_hid += 1
        if data_args.do_predict_task:
            output_pred_file = os.path.join(training_args.output_dir, f"task_disc_pred.json")
            with open(output_pred_file,"w") as F:
                json.dump(all_task_pred ,F)
        if data_args.vis_hidden:
            all_hidden_states = np.array([i.numpy() for i in all_hidden_states])
            vis_hidden(all_hidden_states,all_y,data_args.vis_hidden_file, ALL_TASK_NAMES)
    return eval_results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
