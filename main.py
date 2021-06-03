from prepare import setup
import pdb

source_file = "./train_source.txt"
target_file = "./train_target.txt"
batch_size = 64
max_len = 100

train_data, val_data, voca_x, voca_y = setup(source_file, target_file, max_len)
EOS = voca_x['<eos>']
BOS = voca_x['<bos>']
PAD = voca_x['<pad>']

for i in range(len(train_data)//batch_size):
    train_x, train_y = train_data.get_batch(i, batch_size)
pdb.set_trace()
print(train_x)
