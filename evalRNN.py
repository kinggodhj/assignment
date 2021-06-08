import sys
import argparse
import time
import math
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
import random
import pdb

from rnnModel import EncoderRNN, AttnDecoderRNN
from prepare import setupRNN

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--bi', type=bool, default=False)

args = parser.parse_args()
MAX_LEN = args.max_len
EMB_SIZE = args.emb_size
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs

if args.bi == True:
#    sys.stdout = open('./generated/Bimodel%s%s'%(NUM_EPOCHS, EMB_SIZE), 'w')
    PATH1 = './attn/Biencoder%s%s.pkt'%(NUM_EPOCHS, EMB_SIZE)
    PATH2 = './attn/Bidecoder%s%s.pkt'%(NUM_EPOCHS, EMB_SIZE)
else:
#    sys.stdout = open('./generated/model%s%s'%(NUM_EPOCHS, EMB_SIZE), 'w')
    PATH1 = './attn/encoder%s%s.pkt'%(NUM_EPOCHS, EMB_SIZE)
    PATH2 = './attn/decoder%s%s.pkt'%(NUM_EPOCHS, EMB_SIZE)

def evaluate(val_iter, encoder, decoder, max_length=MAX_LEN):
    encoder.eval()
    decoder.eval()
    losses = 0
    for idx, (src, tgt) in enumerate(val_iter):
        if src.size(0) != BATCH_SIZE:
            continue
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        src = torch.transpose(src, 0, 1)
        tgt = torch.transpose(tgt, 0, 1)

        target_length = tgt.size(0)

        encoder_hidden = encoder.initHidden()

        encoder_outputs, encoder_hidden = encoder(src, encoder_hidden)
        
        if len(encoder_hidden) > 1:
            bidir = torch.cat([encoder_hidden[0], encoder_hidden[1]], dim=-1)
            proj = nn.Linear(EMB_SIZE * 2, EMB_SIZE).to(DEVICE)
            encoder_hidden = proj(bidir).to(DEVICE)
            encoder_hidden = encoder_hidden.view(1, BATCH_SIZE, EMB_SIZE)
            
            proj2 = nn.Linear(EMB_SIZE * 2, EMB_SIZE).to(DEVICE)
            encoder_outputs = proj2(encoder_outputs)
        else:
            encoder_hidden = encoder_hidden[-1]
            encoder_hidden = encoder_hidden.unsqueeze(0)

        decoder_input = torch.ones(BATCH_SIZE).fill_(BOS).to(DEVICE).long()
        decoder_input = decoder_input.view(-1)

        decoder_hidden = encoder_hidden

        decoder_outputs = []
        logits = [] 
        for _ in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(decoder_input, decoder_hidden, encoder_outputs)
            logits.append(decoder_output)
            topv, topi = decoder_output.topk(1)
            decoder_input = topi.squeeze().detach()  # detach from history as input
            decoder_outputs.append(decoder_input)
            decoder_input = decoder_input.view(-1)
            if decoder_input.item() == EOS:
                break
        
        #logits = torch.stack(logits)
        #logits = logits.reshape(-1, logits.size(-1))

        tgt_item = []
       
        for di in range(target_length):
            if tgt[di].item() == EOS:
                break
            tgt_item.append(tgt[di])
        decoder_outputs = torch.stack(decoder_outputs)
        tgt_c = torch.stack(tgt_item)

        print(decoder_outputs.tolist())            

if __name__ == "__main__":
    source_file = "./train_x.0.txt"
    target_file = "./train_y.0.txt"
    
    val_source_file = "./test_source.txt"
    val_target_file = "./test_target.txt"

    train_data, voca_x, voca_y = setupRNN(source_file, target_file, MAX_LEN)
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_data.get_batch)

    val_data, _, _ = setupRNN(val_source_file, val_target_file, MAX_LEN, voca_x, voca_y)
    val_iter = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=val_data.get_batch)

    EOS = voca_x['<eos>']
    BOS = voca_x['<bos>']
    PAD = voca_x['<pad>']

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    encoder = EncoderRNN(len(voca_x), BATCH_SIZE, EMB_SIZE, args.bi).to(DEVICE)
    attn_decoder = AttnDecoderRNN(BATCH_SIZE, EMB_SIZE, len(voca_y), dropout_p=0.1).to(DEVICE)

    encoder.load_state_dict(torch.load(PATH1))
    attn_decoder.load_state_dict(torch.load(PATH2))

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001,  betas=(0.9, 0.98), eps=1e-9)
    decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=0.0001,  betas=(0.9, 0.98), eps=1e-9)
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)

    evaluate(val_iter, encoder, attn_decoder)
