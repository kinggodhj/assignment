from collections import Counter
from sklearn.utils import shuffle
from torchtext.vocab import Vocab
import pdb

from data import SequenceDataset


def filterPair(d1, d2, max_len):
    return len(d1) < max_len and  len(d2) < max_len 

def filterPairs(d1, d2, max_len):
    filter_x = []
    filter_y = []
    for x, y in zip(d1, d2):
        if filterPair(x, y, max_len):
            filter_x.append(x)
            filter_y.append(y)
    return filter_x, filter_y

def read_file(path):
    f = open(path, 'r')
    lines = f.readlines()
    data = []
    for line in lines:
        data.append(list(map(int, line.rstrip().split(' '))))
    f.close()

    return data

#def build_vocab(path):
def build_vocab(data):
    counter = Counter(sum(data, []))   
    vocab = Vocab(counter, specials = ['<unk>', '<pad>', '<bos>', '<eos>'])
    return vocab    

def setup(path1, path2, max_len):
    x = read_file(path1)
    y = read_file(path2)
    x, y = filterPairs(x, y, max_len)
    
    x, y = shuffle(x, y)
    
    train_x = x[:int(len(x)*0.8)]
    val_x = x[int(len(x)*0.8):]
    
    train_y = y[:int(len(x)*0.8)]
    val_y = y[int(len(x)*0.8):]

    voca_x = build_vocab(train_x)
    voca_y = build_vocab(train_y)
    
    train_data = SequenceDataset(train_x, train_y, max_len, voca_x, voca_y)
    val_data = SequenceDataset(val_x, val_y, max_len, voca_x, voca_y)

    return train_data, val_data, voca_x, voca_y
