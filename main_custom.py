import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import math
from nltk.translate.bleu_score import sentence_bleu
import time
import pdb

from model_custom import Seq2SeqTransformer, create_mask, greedy_decode
from prepare import setup

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--emb_size', type=int, default=32)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--ffd_dim', type=int, default=32)
parser.add_argument('--num_encoder_layers', type=int, default=2)
parser.add_argument('--num_decoder_layers', type=int, default=2)
parser.add_argument('--epochs', type=int, default=25)

args = parser.parse_args()

MAX_LEN = args.max_len
EMB_SIZE = args.emb_size
NHEAD = args.nhead
FFN_HID_DIM = args.ffd_dim
BATCH_SIZE = args.batch_size
NUM_ENCODER_LAYERS = args.num_encoder_layers
NUM_DECODER_LAYERS = args.num_decoder_layers
NUM_EPOCHS = args.epochs

writer = SummaryWriter('./runs/%s%s%s%s'%(EMB_SIZE, NUM_ENCODER_LAYERS, NUM_EPOCHS, MAX_LEN))

def train(model, train_iter, optimizer):
    model.train()
    losses = 0
    for idx, (src, tgt) in enumerate(train_iter):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:,:]
        #pdb.set_trace()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return model, (losses / len(train_iter))


def evaluate(model, val_iter):
    model.eval()
    losses = 0
    #for i in range(len(val_data) // BATCH_SIZE):
    #    src, tgt = val_data.get_batch(i, BATCH_SIZE)
    for idx, (src, tgt) in enumerate(val_iter):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:,:]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(val_iter)

def get_bleu(model, val_iter_p):
    model.eval()
    bleu = 0
    for idx, (src, tgt) in enumerate(val_iter_p):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD)

        ys = greedy_decode(model, src, src_mask, MAX_LEN, BOS, EOS)
    
        target = tgt.tolist()
        target = sum(target, [])
        target = list(map(str, target))
        
        pred = ys.tolist()
        pred = sum(pred, [])
        pred = list(map(str, pred))
        bleu += sentence_bleu(target, pred)

    return bleu / len(val_iter)


if __name__ == "__main__":
    source_file = "./train_x.0.txt"
    target_file = "./train_y.0.txt"
    
    val_source_file = "./train_x.1.txt"
    val_target_file = "./train_y.1.txt"

    train_data, voca_x, voca_y = setup(source_file, target_file, MAX_LEN)
    
    train_iter = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=train_data.get_batch)
    
    val_data, _, _ = setup(val_source_file, val_target_file, MAX_LEN, voca_x, voca_y)
    val_iter = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=val_data.get_batch)
    
    #val_iter_p = DataLoader(val_data, batch_size=1, shuffle=True, collate_fn=val_data.get_batch)
    
    SRC_VOCAB_SIZE = len(voca_x)
    TGT_VOCAB_SIZE = len(voca_y)

    EOS = voca_x['<eos>']
    BOS = voca_x['<bos>']
    PAD = voca_x['<pad>']
    
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NHEAD, EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)    
    

    transformer = transformer.to(DEVICE)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)
    #pdb.set_trace()

    for epoch in range(1, NUM_EPOCHS+1):
        start_time = time.time()
        transformer, train_loss = train(transformer, train_iter, optimizer)
        train_ppl = math.exp(train_loss)
        end_time = time.time()
        val_loss = evaluate(transformer, val_iter)
        val_ppl = math.exp(val_loss)
        #bleu = get_bleu(transformer, val_iter_p)
        print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, PPL: {train_ppl:.3f}, Val loss: {val_loss:.3f}, PPL: {val_ppl:.3f})  " f"Epoch time = {(end_time - start_time):.3f}s"))
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('PPL/train', train_ppl, epoch)
        writer.add_scalar('PPL/Val', val_ppl, epoch)
        #writer.add_scalar('BLEU/Val', bleu, epoch)
    torch.save(transformer.state_dict(), './premodel/model%s%s%s%s.pkt'%(NUM_EPOCHS, EMB_SIZE, NHEAD, NUM_ENCODER_LAYERS))
