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

from rnnModel import EncoderRNN, AttnDecoderRNN
from prepare import setupRNN

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
teacher_forcing_ratio = 0.5

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--emb_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)

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
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        src = src.view(-1)
        tgt = tgt.view(-1)
        encoder_hidden = encoder.initHidden()

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        input_length = src.size(0)
        target_length = tgt.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(src[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[BOS]], device=DEVICE)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss = loss_fn(decoder_output, tgt[di].view(-1))
                losses += loss.item()
                decoder_input = tgt[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss = loss_fn(decoder_output, tgt[di].view(-1))
                losses += loss.item()
                if decoder_input.item() == EOS:
                    break

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

    return losses / len(train_iter)

def evaluate(val_iter, encoder, decoder, max_length=MAX_LEN):
    encoder.train()
    decoder.train()
    losses = 0
    for idx, (src, tgt) in enumerate(val_iter):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        src = src.view(-1)
        tgt = tgt.view(-1)
        
        encoder_hidden = encoder.initHidden()
        input_length = src.size(0)
        target_length = tgt.size(0)

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=DEVICE)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(src[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[BOS]], device=DEVICE)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss = loss_fn(decoder_output, tgt[di].view(-1))
                losses += loss.item()
                decoder_input = tgt[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss = loss_fn(decoder_output, tgt[di].view(-1))
                losses += loss.item()
                if decoder_input.item() == EOS:
                    break

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

    encoder = EncoderRNN(len(voca_x), EMB_SIZE).to(DEVICE)
    attn_decoder = AttnDecoderRNN(EMB_SIZE, len(voca_y), dropout_p=0.1).to(DEVICE)

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.0001,  betas=(0.9, 0.98), eps=1e-9)
    decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=0.0001,  betas=(0.9, 0.98), eps=1e-9)
    
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)

    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        train_loss = train(train_iter, encoder, attn_decoder, encoder_optimizer, decoder_optimizer)
        train_ppl = math.exp(train_loss)

        val_loss = evaluate(val_iter, encoder, attn_decoder)
        val_ppl = math.exp(val_loss)

        print("Epoch:", epoch, "Train loss:", train_loss, "PPL:", train_ppl, "Val loss:", val_loss, "PPL:", val_ppl, "Epoch time =", (end_time - start_time))
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('PPL/train', train_ppl, epoch)
        writer.add_scalar('PPL/Val', val_ppl, epoch)

    torch.save(encoder.state_dict(), './model/encoder%s%s.pkt'%(NUM_EPOCHS, EMB_SIZE))
    torch.save(attn_decoder.state_dict(), './model/decoder%s%s.pkt'%(NUM_EPOCHS, EMB_SIZE))
