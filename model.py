import torch.nn as nn
from transformers import BertModel, BertConfig

class BertMultiLabelClassifier(nn.Module):
    def __init__(self, model_name, num_labels, dropout=0.1):
        super(BertMultiLabelClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_labels)
        )

    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
        return {'loss': loss, 'logits': logits} if loss is not None else logits

def freeze_base_layers(model):
    for param in model.bert.parameters():
        param.requires_grad = False
    # Optionally, unfreeze the last few layers if desired
    # for param in model.bert.encoder.layer[-2:].parameters():
    #     param.requires_grad = True
