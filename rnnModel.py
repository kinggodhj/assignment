import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import pdb
DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

class EncoderRNN(nn.Module):
    def __init__(self, input_size, batch_size, hidden_size, bi=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.bidir = bi
        self.gru = nn.GRU(hidden_size, hidden_size, bidirectional=self.bidir)

    def forward(self, input, input_l, hidden):
        embedded = self.embedding(input)
        output, hidden = self.gru(embedded, hidden)

        return output, hidden

    def initHidden(self):
        if self.bidir:
            return torch.zeros(2, self.batch_size, self.hidden_size, device=DEVICE)
        return torch.zeros(1, self.batch_size, self.hidden_size, device=DEVICE)

class AttnDecoderRNN(nn.Module):
    def __init__(self, batch_size, hidden_size, output_size, dropout_p=0.1, max_length=100):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
       
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), torch.transpose(encoder_outputs, 0, 1))
        attn_applied = attn_applied.view(self.batch_size, self.hidden_size)
        
        output = torch.cat((embedded, attn_applied), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output = output.view(1, self.batch_size, self.hidden_size)
        
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=DEVICE)

class DecoderRNN(nn.Module):
    def __init__(self, batch_size, hidden_size, output_size, dropout_p=0.1, max_length=100):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        
        embedded = embedded.view(-1, self.batch_size, self.hidden_size)

        output, hidden = self.gru(embedded, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.batch_size, self.hidden_size, device=DEVICE)
