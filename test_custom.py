import sys
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

from model_custom import Seq2SeqTransformer, create_mask, greedy_decode, greedy_decode2
from prepare import build_vocab, setup

DEVICE = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--max_len', type=int, default=100)
parser.add_argument('--emb_size', type=int, default=64)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--ffd_dim', type=int, default=64)
parser.add_argument('--num_encoder_layers', type=int, default=1)
parser.add_argument('--num_decoder_layers', type=int, default=1)
parser.add_argument('--path', type=str, default='./model/')
parser.add_argument('--epochs', type=int, default=50)


args = parser.parse_args()

MAX_LEN = args.max_len
EMB_SIZE = args.emb_size
NHEAD = args.nhead
FFN_HID_DIM = args.ffd_dim
NUM_ENCODER_LAYERS = args.num_encoder_layers
NUM_DECODER_LAYERS = args.num_decoder_layers
NUM_EPOCHS = args.epochs

PATH = args.path

sys.stdout = open('./generated/batch/%spremodel%s%s%s'%(args.batch_size, NUM_EPOCHS, EMB_SIZE, NUM_ENCODER_LAYERS), 'w')

def get_bleu(model, vocab, test_iter):
    model.eval()
    bleu = 0
    losses = 0
    for idx, (src, tgt) in enumerate(test_iter):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD)

        tgt_out = tgt[1:,:]
        
        ys = greedy_decode(model, src, src_mask, MAX_LEN, BOS, EOS)
        logits = greedy_decode2(model, src, src_mask, tgt_out.size(0), BOS, EOS)    
        
        loss_fn = torch.nn.CrossEntropyLoss()
        
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        
        losses += loss.item()

        target = tgt.tolist()
        target = sum(target, [])
        target = list(map(str, target))
        target = target[1:-1]

        pred = ys.tolist()
        pred = sum(pred, [])
        
        pred = pred[1:-1]
        
        generated = []
        for p in pred:
            generated.append(vocab.itos[p])

        print(*generated, sep=" ")


def evaluate(model, test_iter):
    model.eval()
    losses = 0
    for idx, (src, tgt) in enumerate(test_iter):
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input, PAD)

        logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)
        tgt_out = tgt[1:,:]
        loss_fn = torch.nn.CrossEntropyLoss()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()
    return losses / len(test_iter)

if __name__ == "__main__":
    
    train_data, voca_x, voca_y = setup("./train_x.0.txt", "./train_y.0.txt", MAX_LEN)
    
    source_file = "./test_source.txt"
    target_file = "./test_target.txt"

    test_data, _, _ = setup(source_file, target_file, MAX_LEN)
    
    train_iter = DataLoader(train_data, batch_size=1, shuffle=True, collate_fn=train_data.get_batch)
    
    test_iter = DataLoader(test_data, batch_size=1, shuffle=True, collate_fn=test_data.get_batch)
    
    SRC_VOCAB_SIZE = len(voca_x)
    TGT_VOCAB_SIZE = len(voca_y)

    EOS = voca_x['<eos>']
    BOS = voca_x['<bos>']
    PAD = voca_x['<pad>']

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, NHEAD, EMB_SIZE, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

    transformer = transformer.to(DEVICE)
    transformer.load_state_dict(torch.load(PATH))

    #train_bleu = get_bleu(transformer, train_iter)
    get_bleu(transformer, voca_y, test_iter)
    #loss = evaluate(transformer, test_iter)
    #ppl = math.exp(loss)
    
    #train_loss = evaluate(transformer, train_iter)
    #train_ppl = math.exp(train_loss)

    #print('Train PPL:', train_ppl, 'Test PPL:', ppl)
