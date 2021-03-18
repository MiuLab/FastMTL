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
#TEMP_MODIFY: grad_disc requires grad
def prediction_step(
    self,
    model: nn.Module,
    inputs: Dict[str, Union[torch.Tensor, Any]],
    prediction_loss_only: bool,
    ignore_keys: Optional[List[str]] = None,
) -> Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Perform an evaluation step on :obj:`model` using obj:`inputs`.
    Subclass and override to inject custom behavior.
    Args:
        model (:obj:`nn.Module`):
            The model to evaluate.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.
        prediction_loss_only (:obj:`bool`):
            Whether or not to return the loss only.
        ignore_keys (:obj:`Lst[str]`, `optional`):
            A list of keys in the output of your model (if it is a dictionary) that should be ignored when
            gathering predictions.
    Return:
        Tuple[Optional[float], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss, logits and
        labels (each being optional).
    """
    has_labels = all(inputs.get(k) is not None for k in self.label_names)
    inputs = self._prepare_inputs(inputs)
    if ignore_keys is None:
        if hasattr(self.model, "config"):
            ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
        else:
            ignore_keys = []

    # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
    if has_labels:
        labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
        if len(labels) == 1:
            labels = labels[0]
    else:
        labels = None

    #TEMP_MODIFY: conditonally set no_grad
    with torch.set_grad_enabled(self.model.do_predict_grad):
    #END_TEMP
        if has_labels:
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            loss = loss.mean().detach()
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"])
            else:
                logits = outputs[1:]
        else:
            loss = None
            if self.use_amp:
                with autocast():
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            if isinstance(outputs, dict):
                logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
            else:
                logits = outputs
            # TODO: this needs to be fixed and made cleaner later.
            if self.args.past_index >= 0:
                self._past = outputs[self.args.past_index - 1]

    if prediction_loss_only:
        return (loss, None, None)

    logits = nested_detach(logits)
    if len(logits) == 1:
        logits = logits[0]

    return (loss, logits, labels)


Trainer.get_train_dataloader = get_train_dataloader

#TEMP_MODIFY: add custom function to trainer
Trainer.prediction_step = prediction_step
#END_TEMP
