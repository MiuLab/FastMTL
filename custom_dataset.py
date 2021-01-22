import torch
import collections
def _get_train_sampler(dataset) -> Optional[torch.utils.data.sampler.Sampler]:
    if isinstance(dataset, torch.utils.data.IterableDataset) or not isinstance(
        dataset, collections.abc.Sized
    ):
        return None
    elif is_torch_tpu_available():
        return get_tpu_sampler(dataset)
    else:
        return (
            RandomSampler(dataset)
            if self.args.local_rank == -1
            else DistributedSampler(dataset)
        )
class MergeDataset(Dataset):
    def __init__(self, args, datasets_list):
        self.args = args
        self.num_datasets = len(datasets_list)
        self.dataloader_list = []
        self.iter_list = []
        for dataset in datasets_list:
            train_sampler = _get_train_sampler(dataset)
            # The dataloader in the trainer will get batchsize=1
            dataloader = DataLoader(
                dataset,
                batch_size=self.args.train_batch_size,
                sampler=train_sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
            )
            self.dataloader_list.append(dataloader)
            self.iter_list.append(iter(dataloader))
        #data_idx range
        self.dataloader_range = []
        cnt = 0
        for d_l in dataloader_list:
            self.dataloader_range.append(range(cnt,cnt+len(d_l)))
            cnt+=len(d_l)

    def __len__(self):
        # len is the sum of max iter of all dataloader
        return sum([len(d_l) for d_l in self.dataloader_list])

    def __getitem__(self, idx):
        d_l_id = None
        for i, d_l_range in enumerate(self.dataloader_range):
            if idx in d_l_range:
                d_l_id = i 
        # random sample this dataloader

        inputs = iterator.next()
        print(inputs)
        return inputs
