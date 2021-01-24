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
        #MODIFY --> change classifier to classifier list
        #self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier_list = {}
        for task_name in self.all_task_names:
            self.classifier_list[task_name] = nn.Linear(config.hidden_size, self.num_labels_dict[task_name]) 

        self.init_weights()

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

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        #MODIFY --> use correspond classifier
        #logits = self.classifier(pooled_output)
        logits = self.classifier_list[task_name](pooled_output)

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

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


