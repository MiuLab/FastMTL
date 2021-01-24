from transformers.trainer import *
import logging
def _get_train_sampler(args,dataset):
    # Baically, only the random sampler will be run
    if isinstance(dataset, torch.utils.data.IterableDataset) or not isinstance(
        dataset, collections.abc.Sized
    ):
        return None
    elif is_torch_tpu_available():
        return get_tpu_sampler(dataset)
    else:
        return (
            RandomSampler(dataset)
            if args.local_rank == -1
            else DistributedSampler(dataset)
        )
class MergeDataset(Dataset):
    def __init__(self, args, datasets_list, all_task_names):
        self.args = args
        self.num_datasets = len(datasets_list)
        self.all_task_names = all_task_names
        self.datasets_list = datasets_list
        #set data_collator
        self.data_collator = default_data_collator
        #set dataloader and iter
        self.init_loaders()
        #data_idx range
        self.dataloader_range = []
        cnt = 0
        for d_l in self.dataloader_list:
            self.dataloader_range.append(range(cnt,cnt+len(d_l)))
            cnt+=len(d_l)

    def __len__(self):
        # len is the sum of max iter of all dataloader
        return sum([len(d_l) for d_l in self.dataloader_list])

    def __getitem__(self, idx):
        self.cnt_iter +=1
        d_l_id = None
        for i, d_l_range in enumerate(self.dataloader_range):
            if idx in d_l_range:
                d_l_id = i 
        # random sample this dataloader
        inputs = self.iter_list[d_l_id].next()
        #add task_id key and value, del [idx]? not sure why we get idx
        inputs["task_name"] = self.all_task_names[d_l_id]
        del inputs["idx"]
        #check if one epoch is done
        if self.cnt_iter == self.__len__():
            self.init_loaders()
        return inputs

    # Init loaders and iters
    def init_loaders(self):
        print("------------------------- init Loaders -----------------------")
        #reset self.cnt_iter
        self.cnt_iter = 0
        self.dataloader_list = []
        self.iter_list = []
        for dataset in self.datasets_list:
            train_sampler = _get_train_sampler(self.args, dataset)
            # The dataloader in the trainer will get batchsize=1
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.real_train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
            )
            self.dataloader_list.append(dataloader)
            self.iter_list.append(iter(dataloader))


#Data collator for trainer(sample 1 example each iter)
class CustomDataCollator:
    def __init__(self):
        print("Init custom Dataloader for trainer")

    def __call__(self, examples: List[dict]):
        return examples[0]


