import pandas as pd
import pickle

def make_datasets(file,target_counts,seq_counts,isSave = True):

    #file_path = 'input/u.data'
    file_path = file
    names = ['user', 'item', 'rateing', 'timestamps']
    data = pd.read_csv(file_path, header=None, sep='\t', names=names)

    # ReMap item ids
    item_unique = data['item'].unique().tolist()
    item_map = dict(zip(item_unique, range(1,len(item_unique) + 1)))
    item_map[-1] = 0
    all_item_count = len(item_map)
    data['item'] = data['item'].apply(lambda x: item_map[x])

    # ReMap usr ids
    user_unique = data['user'].unique().tolist()
    user_map = dict(zip(user_unique, range(1, len(user_unique) + 1)))
    user_map[-1] = 0
    all_user_count = len(item_map)
    data['user'] = data['user'].apply(lambda x: user_map[x])

    # Get user session
    data = data.sort_values(by=['user','timestamps']).reset_index(drop=True)

    # 生成用户序列
    user_sessions = data.groupby('user')['item'].apply(lambda x: x.tolist()) \
        .reset_index().rename(columns={'item': 'item_list'})

    train_users = []
    train_seqs = []
    train_targets = []

    test_users = []
    test_seqs = []
    test_targets = []

    user_all_items = {}

    for index, row in user_sessions.iterrows():
        user = row['user']
        items = row['item_list']
        user_all_items[user] = items

        for i in range(seq_counts,len(items) - target_counts):
            targets = items[i:i+target_counts]
            seqs = items[max(0,i - seq_counts):i]

            train_users.append(user)
            train_seqs.append(seqs)
            train_targets.append(targets)

        last_item = [items[-1],0,0]
        last_seq = items[-1 - seq_counts:-1]
        test_users.append(user)
        test_seqs.append(last_seq)
        test_targets.append(last_item)


    train = pd.DataFrame({'user':train_users,'seq':train_seqs,'target':train_targets})

    test = pd.DataFrame({'user': test_users, 'seq': test_seqs, 'target': test_targets})

    if isSave:

        train.to_csv('input/train.csv', index=False)
        test.to_csv('input/test.csv', index=False)
        with open('input/info.pkl','wb+') as f:
            pickle.dump(user_all_items,f,pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_user_count,f,pickle.HIGHEST_PROTOCOL)
            pickle.dump(all_item_count, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(user_map, f, pickle.HIGHEST_PROTOCOL)
            pickle.dump(item_map, f, pickle.HIGHEST_PROTOCOL)

    return train,test,\
           user_all_items,all_user_count,\
           all_item_count,user_map,item_map




if __name__ == '__main__':
    make_datasets(3,5)



'''

src data

196	242	3	881250949
186	302	3	891717742
22	377	1	878887116
244	51	2	880606923
166	346	1	886397596
298	474	4	884182806
115	265	2	881171488
253	465	5	891628467
305	451	3	886324817
6	86	3	883603013
62	257	2	879372434
286	1014	5	879781125
200	222	5	876042340
210	40	3	891035994
224	29	3	888104457



train_data
                          seq            target  user
0     [1, 290, 492, 381, 752]    [467, 523, 11]     1
1   [290, 492, 381, 752, 467]    [523, 11, 673]     1
2   [492, 381, 752, 467, 523]   [11, 673, 1046]     1
3    [381, 752, 467, 523, 11]  [673, 1046, 650]     1
4    [752, 467, 523, 11, 673]  [1046, 650, 378]     1
5   [467, 523, 11, 673, 1046]   [650, 378, 180]     1
6   [523, 11, 673, 1046, 650]   [378, 180, 390]     1
7   [11, 673, 1046, 650, 378]   [180, 390, 666]     1
8  [673, 1046, 650, 378, 180]   [390, 666, 513]     1
9  [1046, 650, 378, 180, 390]   [666, 513, 432]     1

test_data
                            seq       target  user
0    [633, 657, 1007, 948, 364]  [522, 0, 0]     1
1       [26, 247, 49, 531, 146]   [32, 0, 0]     2
2      [459, 477, 369, 770, 15]  [306, 0, 0]     3
3  [1093, 946, 1101, 690, 1211]  [526, 0, 0]     4
4     [732, 266, 669, 188, 253]  [986, 0, 0]     5
5      [410, 446, 104, 782, 96]   [26, 0, 0]     6
6     [146, 817, 536, 694, 186]  [525, 0, 0]     7
7      [395, 669, 281, 289, 98]  [731, 0, 0]     8
8     [588, 671, 369, 292, 304]  [250, 0, 0]     9
9        [472, 222, 82, 716, 8]  [131, 0, 0]    10


'''












