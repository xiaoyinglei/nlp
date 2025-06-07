import pandas as pd
import torch.nn as nn
from transformers import AutoConfig, XLMRobertaConfig, AutoTokenizer
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.roberta.modeling_roberta import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
import torch

# 必须提前定义
tags = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
index2tag = {i: tag for i, tag in enumerate(tags)}
tag2index = {tag: i for i, tag in enumerate(tags)}

xlmr_model_name = "xlm-roberta-base"
xlmr_config = AutoConfig.from_pretrained(
    xlmr_model_name,
    num_labels=len(tags),
    id2label=index2tag,
    label2id=tag2index
)

class XLMRobertaForTokenClassification(RobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, **kwargs)
        sequence_output = self.dropout(outputs[0])
        logits = self.classifier(sequence_output)
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        return TokenClassifierOutput(loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)

# 加载 tokenizer 和模型
tokenizer = AutoTokenizer.from_pretrained(xlmr_model_name)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

xlmr_model = XLMRobertaForTokenClassification.from_pretrained(
    xlmr_model_name, config=xlmr_config).to(device)

# 测试输入
text = "Jack Sparrow loves New York!"
xlmr_tokens = tokenizer.tokenize(text)
input_ids = tokenizer.encode(text, return_tensors="pt")

df = pd.DataFrame([xlmr_tokens, input_ids[0].numpy()], index=["Tokens", "Input IDs"])
print(df)

