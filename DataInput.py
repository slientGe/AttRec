import pandas as pd
import random
import pickle

class DataIterator:

    def __init__(self,
                 data,
                 batch_size = 128,
                 max_seq_length = 5,
                 neg_count = 1,
                 all_items = None,
                 user_all_items = None,
                 shuffle = True):

        self.data = data
        self.datasize = data.shape[0]
        self.neg_count = neg_count
        self.batch_size = batch_size
        self.msl = max_seq_length
        self.user_all_items = user_all_items
        self.all_items = all_items
        self.shuffle = shuffle
        self.seed = 0
        self.idx=0

    def __iter__(self):
        return self

    def reset(self):
        self.idx = 0
        if self.shuffle:
            self.data= self.data.sample(frac=1).reset_index(drop=True)
            self.seed = self.seed + 1
            random.seed(self.seed)

    def __next__(self):
        if self.idx  >= self.datasize:
            self.reset()
            raise StopIteration

        nums = self.batch_size
        if self.datasize - self.idx < self.batch_size:
            nums  = self.datasize - self.idx

        cur = self.data.iloc[self.idx:self.idx+nums]

        user = cur['user'].values

        #train = pd.DataFrame({'user': train_users, 'seq': train_seqs, 'target': train_targets})

        target = []
        for t in cur['target'].values:
            target.append(t)

        user_seq = []
        sl = []
        for seq in cur['seq'].values:
            user_seq.append(seq)
            sl.append(len(seq))

        neg_seq = []
        for u in cur['user']:
            user_item_set = set(self.all_items) - set(self.user_all_items[u])
            neg_seq.append(random.sample(user_item_set,self.neg_count))

        self.idx += self.batch_size
        return (user, target, user_seq, sl, neg_seq)
