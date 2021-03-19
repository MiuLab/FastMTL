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
        #MODIFY --> for pred_task
        self.do_predict_task=False
        #MODIFY --> For task discriminator
        self.train_task_disc = config.train_task_disc
        self.task_discriminator = nn.Linear(config.hidden_size, config.task_disc_num)

        #TEMP_MODIFY: gradient discriminator
        self.do_predict_grad = False
        self.train_grad_disc = config.train_grad_disc if hasattr(config, "train_grad_disc") else False
        self.grad_discriminator = nn.Linear(sum(p.numel() for p in self.bert.parameters() if p.requires_grad), config.task_disc_num)
        #END_TEMP

        self.name_to_id = {}
        #Use CE for task disc, usually set to False, need to be add to the run_glue.py later...
        self.train_disc_CE = False
        for i, name in enumerate(self.all_task_names):
            self.name_to_id[name] = i

        #MODIFY --> init ignore_list = []
        self.ignore_list = []
        self.init_weights()

    #MODIFY--> MAKE sure things go with the data_args
    def update_args(self, data_args):
        self.train_task_disc = data_args.train_task_disc
        self.vis_hidden = False
        self.do_predict_task=False

        #TEMP_MODIFY: grad_disc
        self.train_grad_disc = data_args.train_grad_disc
        self.do_predict_grad = False
        #END_TEMP

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
        if self.do_predict_task:# return predict task
            sigmoid = torch.nn.Sigmoid()
            return sigmoid(self.task_discriminator(pooled_output))
        logits = self.classifier_dict[task_name](pooled_output)


        #MODIFY --> assign self.num_labels using task_name
        self.num_labels = self.num_labels_dict[task_name]
        loss = None
        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                #TEMP_MODIFY: batch loss should not be reduced when doing grad_disc
                loss_fct = MSELoss(reduction = "none" if self.train_grad_disc or self.do_predict_grad else "mean")
                #END_TEMP
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                #TEMP_MODIFY: same reduction hadle for grad_disc
                loss_fct = CrossEntropyLoss(reduction = "none" if self.train_grad_disc or self.do_predict_grad else "mean")
                #END_TEMP
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if self.weight_loss and labels is not None:
            loss = loss * self.task_loss_weight[self.name_to_id[task_name]]

        #For train_task_disc, joint loss
        if self.train_task_disc and (labels is not None):
            if self.train_disc_CE:
                task_disc_labels = torch.LongTensor([self.name_to_id[task_name]]*logits.size(0)).to(logits.device)
                task_disc_loss_fct = CrossEntropyLoss()
            else:
                task_label_onehot = torch.zeros((pooled_output.size(0),len(self.all_task_names))).to(logits.device)
                task_label_scatter = torch.LongTensor([[self.name_to_id[task_name]]]*logits.size(0)).to(logits.device)
                task_disc_labels = task_label_onehot.scatter_(1, task_label_scatter, 1).to(logits.device)
                task_disc_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.task_disc_pos_weight.to(logits.device))
            task_disc_logits = self.task_discriminator(pooled_output)
            task_disc_loss = task_disc_loss_fct(task_disc_logits, task_disc_labels)
            loss += task_disc_loss

        #TEMP_MODFIY: grad forward
        if self.train_grad_disc or self.do_predict_grad:
            grads = []
            for i in range(pooled_output.size(0)):
                cat_grad = torch.tensor([]).to(logits.device)
                all_grad = torch.autograd.grad(loss[i], self.bert.parameters(), retain_graph = True if i != pooled_output.size(0) - 1 else False)

                for grad_part in all_grad:
                    cat_grad = torch.cat([cat_grad, grad_part.view(-1)])

                grads.append(cat_grad)

            grads = torch.stack(grads)

            #default is only train the discriminator
            if self.train_grad_disc:
                self.zero_grad()

            if self.do_predict_grad:
                with torch.no_grad():
                    prob_layer = torch.nn.Sigmoid() if not self.train_disc_CE else torch.nn.Softmax(dim = 1)
                    return prob_layer(self.grad_discriminator(grads))

            if self.train_disc_CE:
                grad_disc_labels = torch.LongTensor([self.name_to_id[task_name]]*logits.size(0)).to(logits.device)
                grad_disc_loss_fct = CrossEntropyLoss()
            else:
                grad_label_onehot = torch.zeros((pooled_output.size(0),len(self.all_task_names))).to(logits.device)
                grad_label_scatter = torch.LongTensor([[self.name_to_id[task_name]]]*logits.size(0)).to(logits.device)
                grad_disc_labels = grad_label_onehot.scatter_(1, grad_label_scatter, 1).to(logits.device)
                #temoparily not modify the name "task_disc_pos_weight", but it's not good here
                grad_disc_loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.task_disc_pos_weight.to(logits.device))

            grad_disc_logits = self.grad_discriminator(grads)
            loss = grad_disc_loss_fct(grad_disc_logits, grad_disc_labels)
        #END_TEMP
        
        #MODIFY --> if task in ignore list, ignore it in training, loss = 0
        if task_name in self.ignore_list:
            loss += -loss # set to zero loss
            print("Set Loss to 0: ", loss)



        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


