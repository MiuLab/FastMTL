#import original file
from transformers.models.bert.modeling_bert import *
import logging

# Modified model
# Init --> Config should have num_labels_dict
# Forward --> need to input task_id
class BertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        #MODIFY --> change num_labels to num_labels_dict
        #self.num_labels = config.num_labels
        self.num_labels_dict = config.num_labels_dict
        self.all_task_names = config.all_task_names

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        #MODIFY --> change classifier to classifier dict
        #self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_dict = nn.ModuleDict({})
        for task_name in self.all_task_names:
            self.classifier_dict[task_name] = nn.Linear(config.hidden_size, self.num_labels_dict[task_name]) 
        #MODIFY --> Init a param called task_name for eval & predict to specify which task it is running
        self.task_name = None
        #MODIFY --> for vis_hidden
        self.vis_hidden=False
        #MODIFY --> For task discriminator
        self.train_task_disc = config.train_task_disc
        if self.train_task_disc:
            self.task_discriminator = nn.Linear(config.hidden_size, len(self.all_task_names))
            self.name_to_id = {}
            for i, name in enumerate(self.all_task_names):
                self.name_to_id[name] = i

        self.init_weights()
    #MODIFY--> MAKE sure things go with the data_args
    def update_args(self, data_args):
        self.train_task_disc = data_args.train_task_disc
        self.vis_hidden = data_args.vis_hidden


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        #MODIFY --> add flag to indicate task
        task_name = None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        #MODIFY --> use correspond classifier
        #logits = self.classifier(pooled_output)
        if task_name is None:
            task_name = self.task_name

        pooled_output = outputs[1]
        #For vis hidden
        if self.vis_hidden and (not self.training):#eval or predict
            self.tasks_hs_dict[task_name] += [o.clone().detach().cpu() for o in pooled_output]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier_dict[task_name](pooled_output)


        #MODIFY --> assign self.num_labels using task_name
        self.num_labels = self.num_labels_dict[task_name]
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        #For train_task_disc, joint loss
        if self.train_task_disc and (labels is not None):
            task_disc_labels = torch.LongTensor([self.name_to_id[task_name]]*logits.size(0)).to(logits.device)
            task_disc_logits = self.task_discriminator(pooled_output)
            task_disc_loss_fct = CrossEntropyLoss()
            task_disc_loss = task_disc_loss_fct(task_disc_logits, task_disc_labels)
            loss += task_disc_loss



        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


