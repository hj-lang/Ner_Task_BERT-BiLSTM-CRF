import torch
import torch.nn as nn
from transformers import BertModel
from torchcrf import CRF


class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self, label_size, embedding_dim=768, hidden_dim=256, model_name='bert-base-chinese'):
        super(Bert_BiLSTM_CRF,self).__init__()
        self.label_size = label_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.bert = BertModel.from_pretrained(model_name,return_dict=False)
        self.lstm = nn.LSTM(input_size=self.embedding_dim, hidden_size=self.hidden_dim//2, num_layers=2, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(self.hidden_dim, self.label_size)
        self.crf = CRF(self.label_size, batch_first=True)

    def forward(self, sentence, label, mask, is_test=False):
        with torch.no_grad():
            embeddings, _ = self.bert(sentence)
        lstm_embed, _ = self.lstm(embeddings)
        drop_embed = self.dropout(lstm_embed)
        linear_out = self.linear(drop_embed)
        if is_test:
            decode = self.crf.decode(linear_out, mask)
            return decode
        else:
            loss = -self.crf.forward(linear_out, label, mask, reduction='mean')
            return loss