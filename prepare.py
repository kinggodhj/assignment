from collections import Counter
from sklearn.utils import shuffle
from torchtext.vocab import Vocab
import pdb

from data import SequenceDataset, RNNDataset


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

def setup(path1, path2, max_len, voca_x=None, voca_y=None):
    x = read_file(path1)
    y = read_file(path2)
    x, y = filterPairs(x, y, max_len)
    
    x, y = shuffle(x, y)
   
    if voca_x is not None:
        train_data = SequenceDataset(x, y, max_len, voca_x, voca_y)
    
        return train_data, None, None
    else:
        voca_x = build_vocab(x)
        voca_y = build_vocab(y)

        train_data = SequenceDataset(x, y, max_len, voca_x, voca_y)
    
        return train_data, voca_x, voca_y

def setupRNN(path1, path2, max_len, voca_x=None, voca_y=None):
    x = read_file(path1)
    y = read_file(path2)
    x, y = filterPairs(x, y, max_len)
    
    x, y = shuffle(x, y)
   
    if voca_x is not None:
        train_data = RNNDataset(x, y, max_len, voca_x, voca_y)
    
        return train_data, None, None
    else:
        voca_x = build_vocab(x)
        voca_y = build_vocab(y)

        train_data = RNNDataset(x, y, max_len, voca_x, voca_y)
    
        return train_data, voca_x, voca_y
