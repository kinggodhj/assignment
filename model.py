import math
import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerDecoder, TransformerEncoderLayer, TransformerDecoderLayer

import pdb

from beam import *

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

class Seq2SeqTransformer(nn.Module):
    def __init__(self, num_encoder_layers, num_decoder_layers, num_head, emb_size, src_vocab_size, tgt_vocab_size, dim_feedforward = 512, dropout = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=emb_size, nhead=num_head, dim_feedforward=dim_feedforward)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = TransformerDecoderLayer(d_model=emb_size, nhead=num_head, dim_feedforward=dim_feedforward)
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, trg, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, memory_key_padding_mask):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src, src_mask):
        return self.transformer_encoder(self.positional_encoding(self.src_tok_emb(src)), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.transformer_decoder(self.positional_encoding(self.tgt_tok_emb(tgt)), memory, tgt_mask)

class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0),:])

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size
    def forward(self, tokens):
        #if tokens.long() > self.embedding.size(0):
        #    tokens = 
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def create_mask(src, tgt, pad_idx):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len), device=DEVICE).type(torch.bool)

    src_padding_mask = (src == pad_idx).transpose(0, 1)
    tgt_padding_mask = (tgt == pad_idx).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

def greedy_decode(model, src, src_mask, max_len, start_symbol, EOS_IDX):
    #src = src.to(DEVICE)
    #src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    #ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    # ys: (1, batch)
    ys = torch.ones(1, src.size(-1)).fill_(start_symbol).type(torch.long).to(DEVICE)
   
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(DEVICE).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)) .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        #prob = model.generator(out[:, -1])
        #_, next_word = torch.max(prob, dim = 1)
        #next_word = next_word.item()
        #ys = torch.cat([ys, torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        #if next_word == EOS_IDX:
        #    break


        # prob: seq, batch, len(vocab)
        prob = model.generator(out[-1, :])
        _, next_word = torch.max(prob, dim = -1)
        ys = torch.cat([ys, next_word.view(-1, src.size(-1))], dim=0)
        if next_word.squeeze().item() == EOS_IDX:
            break

    return ys

def greedy_decode2(model, src, src_mask, target_len, start_symbol, EOS_IDX):
    memory = model.encode(src, src_mask)
    ys = torch.ones(1, src.size(-1)).fill_(start_symbol).type(torch.long).to(DEVICE)
   
    logits = [] 
    for i in range(target_len):
        memory = memory.to(DEVICE)
        memory_mask = torch.zeros(ys.shape[0], memory.shape[0]).to(DEVICE).type(torch.bool)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0)) .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)

        # prob: seq, batch, len(vocab)
        prob = model.generator(out[-1, :])
        _, next_word = torch.max(prob, dim = -1)
        ys = torch.cat([ys, next_word.view(-1, src.size(-1))], dim=0)
        logits.append(prob)

    return torch.stack(logits)

