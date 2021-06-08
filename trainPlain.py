import argparse
import time
import math
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import numpy as np
import random
import pdb

from rnnModel import EncoderRNN, DecoderRNN
from prepare import setupRNN

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
teacher_forcing_ratio = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=200)

args = parser.parse_args()

MAX_LEN = args.max_len
EMB_SIZE = args.emb_size
BATCH_SIZE = args.batch_size
NUM_EPOCHS = args.epochs

writer = SummaryWriter('./runs/%s'%(EMB_SIZE))

def train(train_iter, encoder, decoder, encoder_optimizer, decoder_optimizer, max_length=MAX_LEN):
    encoder.train()
    decoder.train()
    losses = 0
    for idx, (src, tgt) in enumerate(train_iter):
        if src.size(0) != BATCH_SIZE:
            continue
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        src = torch.transpose(src, 0, 1)
        tgt = torch.transpose(tgt, 0, 1)

        input_length = src.size(0)
        target_length = tgt.size(0)

        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(src, encoder_hidden)
        
        encoder_hidden = encoder_hidden[-1]
        encoder_hidden = encoder_hidden.unsqueeze(0)

        decoder_input = torch.ones(128).fill_(BOS).to(DEVICE).long()
        decoder_input = decoder_input.view(-1)
        #decoder_input = torch.tensor([[BOS]], device=DEVICE)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = tgt[di]  # Teacher forcing
                decoder_outputs.append(decoder_output)

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                decoder_outputs.append(decoder_output)

        decoder_outputs = torch.stack(decoder_outputs)
        loss = loss_fn(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), tgt.reshape(-1))
        losses += loss.item()
                
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

    return losses / len(train_iter)

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

        input_length = src.size(0)
        target_length = tgt.size(0)

        encoder_hidden = encoder.initHidden()

        encoder_outputs, encoder_hidden = encoder(src, encoder_hidden)
        
        encoder_hidden = encoder_hidden[-1]
        encoder_hidden = encoder_hidden.unsqueeze(0)

        decoder_input = torch.ones(128).fill_(BOS).to(DEVICE).long()
        decoder_input = decoder_input.view(-1)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                decoder_input = tgt[di]  # Teacher forcing
                decoder_outputs.append(decoder_output)

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input
                decoder_outputs.append(decoder_output)

        decoder_outputs = torch.stack(decoder_outputs)
        loss = loss_fn(decoder_outputs.reshape(-1, decoder_outputs.size(-1)), tgt.reshape(-1))
        losses += loss.item()
                
    return losses / len(val_iter)


if __name__ == "__main__":
    source_file = "./train_x.0.txt"
    target_file = "./train_y.0.txt"
    
    val_source_file = "./train_x.1.txt"
    val_target_file = "./train_y.1.txt"

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
    
    encoder = EncoderRNN(len(voca_x), BATCH_SIZE, EMB_SIZE).to(DEVICE)
    attn_decoder = DecoderRNN(BATCH_SIZE, EMB_SIZE, len(voca_y), dropout_p=0.1).to(DEVICE)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001,  betas=(0.9, 0.98), eps=1e-9)
    decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=0.0001,  betas=(0.9, 0.98), eps=1e-9)
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train(train_iter, encoder, attn_decoder, encoder_optimizer, decoder_optimizer)
        train_ppl = math.exp(train_loss)

        end_time = time.time()
        val_loss = evaluate(val_iter, encoder, attn_decoder)
        val_ppl = math.exp(val_loss)

        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, PPL: {train_ppl:.3f}, Val loss: {val_loss:.3f}, PPL: {val_ppl:.3f})  " f"Epoch time = {(end_time - start_time):.3f}s"))
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('PPL/train', train_ppl, epoch)
        writer.add_scalar('PPL/Val', val_ppl, epoch)

        if epoch % 50 == 0 :
            torch.save(encoder.state_dict(), './plain/encoder%s%s.pkt'%(epoch, EMB_SIZE))
            torch.save(attn_decoder.state_dict(), './plain/decoder%s%s.pkt'%(epoch, EMB_SIZE))
