import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import pdb

class SequenceDataset(Dataset):
    #def __init__(self, source_file, target_file, batch_size, max_len, voca1, voca2):
    #    self.source_file = source_file
    #    self.target_file = target_file
    #    self.x = read_file(self.source_file)
    #    self.y = read_file(self.target_file)

    def __init__(self, source, target, max_len, voca1, voca2):
        self.max_len = max_len
        self.x = source
        self.y = target
        self.voca_x = voca1
        self.voca_y = voca2
        self.eos = self.voca_x['<eos>']
        self.bos = self.voca_x['<bos>']
        self.pad = self.voca_x['<pad>']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x= [self.voca_x[word] for word in self.x[idx]]
        y = [self.voca_y[word] for word in self.y[idx]] 
        #x += [self.pad] * (self.max_len - len(x))
        #y += [self.pad] * (self.max_len - len(y))
	
        #x = [self.bos] + x + [self.eos]
        #y = [self.bos] + y + [self.bos]

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
#data = SequenceDataset('./train_source.txt', 'train_target.txt')
#print(data.__len__())
