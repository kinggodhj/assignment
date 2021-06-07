import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pdb

class SequenceDataset(Dataset):
    def __init__(self, source, target, max_len, voca1, voca2):
        self.max_len = max_len
        self.x = source
        self.y = target
        self.voca_x = voca1
        self.voca_y = voca2
        self.eos = self.voca_x['<eos>']
        self.bos = self.voca_x['<bos>']
        self.pad = self.voca_x['<pad>']
        self.unk = self.voca_x['<unk>']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = [self.voca_x[word] for word in self.x[idx]]
        y = [self.voca_y[word] for word in self.y[idx]] 
        
        #x = []
        #for word in self.x[idx]:
        #    if self.voca_x[word] == 0:
        #        x.append(self.unk)
        #    else:
        #        x.append(self.voca_x[word])
        #
        #y = []
        #for word in self.y[idx]:
        #    if self.voca_y[word] == 0:
        #        y.append(self.unk)
        #    else:
        #        y.append(self.voca_y[word])
        #
        return torch.tensor(x), torch.tensor(y)

    '''
    def get_batch(self, idx, batch):
        batch_x = []
        batch_y = []
        for i in range(idx * batch, (idx+1) * batch):
            x, y = self.__getitem__(i)
            batch_x.append(torch.Tensor(x).long())
            batch_y.append(torch.Tensor(y).long())
        batch_x = pad_sequence(batch_x, padding_value=self.pad)
        batch_y = pad_sequence(batch_x, padding_value=self.pad)
        
        return torch.stack(batch_x), torch.stack(batch_y)
    '''

    def get_batch(self, data_batch):
        batch_x, batch_y = [], []
        for (x_item, y_item) in data_batch:
            batch_x.append(torch.cat([torch.tensor([self.bos]), x_item, torch.tensor([self.eos])], dim=0))
            batch_y.append(torch.cat([torch.tensor([self.bos]), y_item, torch.tensor([self.eos])], dim=0))

        batch_x = pad_sequence(batch_x, padding_value=self.pad)
        batch_y = pad_sequence(batch_y, padding_value=self.pad)
        return batch_x, batch_y


class RNNDataset(Dataset):
    def __init__(self, source, target, max_len, voca1, voca2):
        self.max_len = max_len
        self.x = source
        self.y = target
        self.voca_x = voca1
        self.voca_y = voca2
        self.eos = self.voca_x['<eos>']
        self.bos = self.voca_x['<bos>']
        self.pad = self.voca_x['<pad>']
        self.unk = self.voca_x['<unk>']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = [self.voca_x[word] for word in self.x[idx]]
        y = [self.voca_y[word] for word in self.y[idx]] 
        return torch.tensor(x), torch.tensor(y)

    def get_batch(self, data_batch):
        batch_x, batch_y = [], []
        for (x_item, y_item) in data_batch:
            #pad = torch.tensor([self.pad] * (self.max_len - (len(x_item) + 1)))
            #batch_x.append(torch.cat([torch.tensor([self.bos]), x_item, torch.tensor([self.eos]), pad], dim=0))
            #pad = torch.tensor([self.pad] * (self.max_len - (len(y_item) + 1)))
            #batch_y.append(torch.cat([torch.tensor([self.bos]), y_item, torch.tensor([self.eos]), pad], dim=0))
            batch_x.append(torch.cat([x_item, torch.tensor([self.eos])], dim=0))
            batch_y.append(torch.cat([y_item, torch.tensor([self.eos])], dim=0))

        return torch.stack(batch_x), torch.stack(batch_y)
