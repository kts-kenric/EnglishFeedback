import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel, DebertaV2Model
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler

from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ds_train_path = "train.csv"
ds_test_path = "test.csv"


def get_tokenizer(model_name = "microsoft/deberta-v3-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('len(tokenizer)', len(tokenizer)) #128001
    return tokenizer


#will test MultilabelStratifiedKFold afterwards
def make_fold(ds_train_path,ds_test_path):
    df_train = pd.read_csv(ds_train_path)
    df_test = pd.read_csv(ds_test_path)
    NUM_FOLDS = 5

    for f, t in df_train.iterrows():
        df_train.loc[f, "fold"] = f % NUM_FOLDS

    validation_df = df_train[df_train.fold == 0].reset_index(drop=True)
    train_df = df_train[df_train.fold != 0].reset_index(drop=True)
    return validation_df, train_df

# def text_to_token(text,tokenizer):
#
#     encoded = tokenizer(text, add_special_tokens= True, padding=True, truncation=False,  max_length=None)#return_tensors="pt",
#     print(encoded)
#     print("********************************")
#
#     #everything in 1d
#     encoded = tokenizer.encode_plus(
#             text,
#             add_special_tokens=True,
#             return_offsets_mapping=False,
#             max_length=None,#max_token_length,
#             truncation=False
#         ) # return ['input_ids', 'attention_mask', 'offset_mapping']
#     print(encoded)


#input = {'input_ids': token_id, 'attention_mask': token_mask}

class EnglishDataset(Dataset):
    #dataset init
    def __init__(self, df, tokenizers, mode='none'):
        self.mode = mode
        self.df = df
        self.tokenizers = tokenizers
        self.length = len(self.df)

    def __len__(self):
        return self.length
    #get item will iterate through the item eg list or df
    def __getitem__(self, index):
        d = self.df.iloc[index]
        full_text = d.full_text
        encoded = self.tokenizers(full_text, add_special_tokens=True, padding=True, truncation=False, max_length=None)  # return_tensors="pt",
        target = [
            d.cohesion,
            d.syntax,
            d.vocabulary,
            d.phraseology,
            d.grammar,
            d.conventions,
        ]
        batch = {}
        batch['index'] = index
        batch['d'] = d
        batch['token_id'] = torch.LongTensor(encoded['input_ids'])
        batch['token_mask'] = torch.FloatTensor(encoded['attention_mask'])
        batch['target'] = torch.FloatTensor(target)
        return batch

tensor_list = [
    'token_id', 'token_mask', 'target',]


#preprocess batch of data before passing into a pytorch model for inference or training
# batch example
# [ {'token_id': [1, 2, 3, 4], 'token_mask': [1, 1, 1, 1], 'target': 0},
#  {'token_id': [5, 6, 7], 'token_mask': [1, 1, 1], 'target': 1},
#  {'token_id': [8, 9, 10, 11, 12], 'token_mask': [1, 1, 1, 1, 1], 'target': 2}
# ]
def null_collate(batch):
    #empty dictionary
    d = {}
    #iterate the keys ( ['token_id', 'token_mask', 'target'] )
    key = batch[0].keys()
    for k in key:
        #example k-> token_id , v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        v = [b[k] for b in batch]
        #if the key is in target, stack them
        if k in ['target']:
            #torch.stack will stack them
            #eg: x = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]
            # torch.stack(x)
            # will get tensor([[1, 2, 3],
            #         [4, 5, 6],
            #         [7, 8, 9]])
            v = torch.stack(v)
        #save the dictionary with collated output into the new dictionary
        d[k] = v

    #---

    #L is list of the sequence length in the batch
    L = [len(t) for t in d['token_id']]
    #length will get the max out of all L list
    length = max(L)
    #len(d['token_id']) will count the number of token_id for the eg above, it will be 3
    batch_size = len(d['token_id'])

    #create 2 tensor with the batchsize as the size of the sequence
    #basically torch.full will initlaize 2 tensor that is filled with 0 and given the max length of the sequence
    token_id = torch.full((batch_size,length),0) #PAD_id = 0
    token_mask = torch.full((batch_size,length),0)

    for b in range(batch_size):
        #pytorch tensor uses comma for indexing
        #example
        # tensor([[0.3126, 0.3791, 0.3087],
        #         [0.0736, 0.4216, 0.0691]])
        # >> > random[1, 2]
        # tensor(0.0691)
        # >> > random[0, 1]
        # tensor(0.3791)

        #so what this is doing is basically assigning the d[tokenid] into the token_id
        #b is batchslot,
        #:L[b] insert until which length
        token_id[b,:L[b]]=d['token_id'][b]
        token_mask[b,:L[b]]=d['token_mask'][b]
    d['token_id'] = token_id
    d['token_mask'] = token_mask
    #return the dictonary for inputing
    return d

def run_check_dataset():
    #get tokenizer (debertaV2)
    tokenizer = get_tokenizer()
    #create validation and train dataset
    train_df, valid_df = make_fold(ds_train_path, ds_test_path)
    valid_df.to_csv('valid.csv', index=False)
    #Create dataset using torch.utils.data
    dataset = EnglishDataset(train_df, tokenizer)
    print(dataset)
    #debugging purpose
    # for i in range(100):
    #     r = dataset[i]
    #     print(r['index'], '-----------')
    #     #use tensort list because the dataset is in tensor
    #     #check all of the tensor list (3 inside)
    #     for k in tensor_list:
    #         v = r[k]
    #         print(k)
    #         print('\t', 'shape:', v.shape)
    #         print('\t', 'dtype:', v.dtype)
    #         print('\t', 'is_contiguous:', v.is_contiguous())
    #         print('\t', 'min/max:', v.min().item(), '/', v.max().item())
    #         print('\t', 'value:')
    #         print('\t\t', v.reshape(-1)[:8].tolist(), '...')
    #         print('\t\t', v.reshape(-1)[-8:].tolist())
    #     print('')

    # def worker_init_fn(worker_id):
    #     np.random.seed(worker_id)
    loader = DataLoader(
        dataset,
        batch_size = 2,
        num_workers = 0,
        pin_memory = False, #speed up the host to device transfer usually
        sampler = SequentialSampler(dataset),
        # seed for worker process so each process generates a unique seed
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
        collate_fn=null_collate,

    )
    #debug
    print( 'batch_size', loader.batch_size)
    print( 'loader_size', len(loader))
    print( 'dataset_size', len(dataset))
    for t, batch in enumerate(loader):
        if t > 5: break
        print('batch ', t, '===================')
        print('index', batch['index'])
        for k in tensor_list:
            v = batch[k]
            print(k)
            print('\t', 'shape:', v.shape)
            print('\t', 'dtype:', v.dtype)
            print('\t', 'is_contiguous:', v.is_contiguous())
            print('\t', 'value:')
            print('\t\t', v)
        print('')


##test

# test_text = train_df.iloc[0]["full_text"]
# print(test_text)
# print(text_to_token(test_text,tokenizer))
if __name__ == '__main__':
    run_check_dataset()

