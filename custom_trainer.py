from transformers.trainer import *
def get_train_dataloader(self) -> DataLoader:
    """
    Returns the training :class:`~torch.utils.data.DataLoader`.

    Will use no sampler if :obj:`self.train_dataset` does not implement :obj:`__len__`, a random sampler (adapted
    to distributed training if necessary) otherwise.

    Subclass and override this method if you want to inject some custom behavior.
    """
    if self.train_dataset is None:
        raise ValueError("Trainer: training requires a train_dataset.")
    train_sampler = self._get_train_sampler()

    return DataLoader(
        self.train_dataset,
        batch_size=self.args.train_batch_size,
        sampler=train_sampler,
        collate_fn=self.data_collator if self.train_data_collator is None else self.train_data_collator,
        drop_last=self.args.dataloader_drop_last,
        num_workers=self.args.dataloader_num_workers,
    )
Trainer.get_train_dataloader = get_train_dataloader
