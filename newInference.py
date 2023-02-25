import torch.cuda.amp as amp
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoConfig, AutoModel, DebertaV2Model
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
import torch.nn as nn
import torch.nn.functional as F

is_amp = True  # True #False
ds_train_path = "/kaggle/input/feedback-prize-english-language-learning/train.csv"
if mode == 'debug':
    ds_test_path = "/kaggle/input/EnglishFeedbackModel/valid.csv"
elif mode == 'submit':
    ds_test_path = "/kaggle/input/feedback-prize-english-language-learning/test.csv"

num_classes = 6


def time_to_str(t, mode='min'):
    if mode == 'min':
        t = int(t) / 60
        hr = t // 60
        min = t % 60
        return '%2d hr %02d min' % (hr, min)
    elif mode == 'sec':
        t = int(t)
        min = t // 60
        sec = t % 60
        return '%2d min %02d sec' % (min, sec)
    else:
        raise NotImplementedError


def get_tokenizer(model_name='/kaggle/input/debertav3small'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('len(tokenizer)', len(tokenizer))  # 128001
    return tokenizer


class EnglishDataset(Dataset):
    # dataset init
    def __init__(self, df, tokenizers, mode='none'):
        self.mode = mode
        self.df = df
        self.tokenizers = tokenizers
        self.length = len(self.df)

    def __len__(self):
        return self.length

    # get item will iterate through the item eg list or df
    def __getitem__(self, index):
        d = self.df.iloc[index]
        full_text = d.full_text
        encoded = self.tokenizers(full_text, add_special_tokens=True, padding=True, truncation=False,
                                  max_length=None)  # return_tensors="pt",
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
    'token_id', 'token_mask', 'target', ]


def null_collate(batch):
    # empty dictionary
    d = {}
    # iterate the keys ( ['token_id', 'token_mask', 'target'] )
    key = batch[0].keys()
    for k in key:
        # example k-> token_id , v = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        v = [b[k] for b in batch]
        # if the key is in target, stack them
        if k in ['target']:
            # torch.stack will stack them
            # eg: x = [torch.tensor([1, 2, 3]), torch.tensor([4, 5, 6]), torch.tensor([7, 8, 9])]
            # torch.stack(x)
            # will get tensor([[1, 2, 3],
            #         [4, 5, 6],
            #         [7, 8, 9]])
            v = torch.stack(v)
        # save the dictionary with collated output into the new dictionary
        d[k] = v

    # ---

    # L is list of the sequence length in the batch
    L = [len(t) for t in d['token_id']]
    # length will get the max out of all L list
    length = max(L)
    # len(d['token_id']) will count the number of token_id for the eg above, it will be 3
    batch_size = len(d['token_id'])

    # create 2 tensor with the batchsize as the size of the sequence
    # basically torch.full will initlaize 2 tensor that is filled with 0 and given the max length of the sequence
    token_id = torch.full((batch_size, length), 0)  # PAD_id = 0
    token_mask = torch.full((batch_size, length), 0)

    for b in range(batch_size):
        # pytorch tensor uses comma for indexing
        # example
        # tensor([[0.3126, 0.3791, 0.3087],
        #         [0.0736, 0.4216, 0.0691]])
        # >> > random[1, 2]
        # tensor(0.0691)
        # >> > random[0, 1]
        # tensor(0.3791)

        # so what this is doing is basically assigning the d[tokenid] into the token_id
        # b is batchslot,
        #:L[b] insert until which length
        token_id[b, :L[b]] = d['token_id'][b]
        token_mask[b, :L[b]] = d['token_mask'][b]
    d['token_id'] = token_id
    d['token_mask'] = token_mask
    # return the dictonary for inputing
    return d


def criterion_mcrmse(predict, truth):
    diff = (predict - truth) ** 2
    loss = diff.mean(0)
    loss = torch.sqrt(loss)
    return loss


class Net(nn.Module):
    def __init__(self, num_classes, model_name="/kaggle/input/debertav3small"):
        super().__init__()
        self.output_type = ['inference', 'loss']
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.predict = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, 64),
            nn.Linear(64, num_classes),

        )
        # what i need config file,
        # model pretrained

    def forward(self, batch):
        token_id = batch['token_id']
        token_mask = batch['token_mask']
        model_out = self.transformer(token_id, token_mask)
        x = model_out.last_hidden_state[:, 0]
        out = self.predict(x)
        output = {}
        if 'loss' in self.output_type:
            output['mse_loss'] = F.mse_loss(out, batch['target'])
            output['l1_loss'] = F.smooth_l1_loss(out, batch['target'])
            output['mcrmse_loss'] = criterion_mcrmse(out, batch['target'])

        if 'inference' in self.output_type:
            output['predict'] = out
        return output


def do_local_evaluate(submit_df, truth_df):
    assert (submit_df['text_id'].equals(truth_df['text_id']))
    predict = submit_df[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].values
    truth = truth_df[['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions']].values
    d = ((truth - predict) ** 2)
    d = d.mean(axis=0)
    d = d ** 0.5
    score = d.mean(axis=0)
    print("zzz")
    ## ..compute here
    return score


def run_inference():
    initial_checkpoint = "/kaggle/input/EnglishFeedbackModel/00001764.model.pth"

    valid_df = pd.read_csv(ds_test_path)
    tokenizer = get_tokenizer()
    valid_dataset = EnglishDataset(valid_df, tokenizer)

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=4,
        num_workers=0,
        pin_memory=False,  # speed up the host to device transfer usually
        sampler=SequentialSampler(valid_dataset),
        collate_fn=null_collate,
        drop_last=False,
    )

    print('valid_dataset : \n%s\n' % (valid_dataset))
    print('\n')

    net = Net(num_classes).cuda()

    # load the inital check point
    f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
    start_iteration = f['iteration']
    start_epoch = f['epoch']
    state_dict = f['state_dict']
    net.predict.load_state_dict(state_dict, strict=False)

    print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    print('\n')

    valid = dict(
        truth=[],
        predict=[],
        metric=0,
        loss=0,
    )
    valid_num = 0

    net.cuda()
    net.eval()
    net.output_type = ['inference', 'loss']
    # start_timer = time.time()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        for k in tensor_list: batch[k] = batch[k].cuda()

        with torch.no_grad():
            with amp.autocast(enabled=is_amp):
                output = torch.nn.parallel.data_parallel(net, batch)
                loss0 = output['l1_loss'].mean()
                loss1 = output['mcrmse_loss']  # .mean()
        valid_num += batch_size
        valid['loss'] += batch_size * loss0.item()
        valid['metric'] += batch_size * np.square(loss1.data.cpu().numpy())
        valid['truth'].append(batch['target'].data.cpu().numpy())
        valid['predict'].append(output['predict'].data.cpu().numpy())

        ------------------------------------------------
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(time.time() - start_timer, 'sec')),
              end='', flush=True)


print('')
assert (valid_num == len(valid_loader.dataset))

# ----------------------
mse_loss = valid['loss'] / valid_num
mcrmse_loss = valid['metric'] / valid_num
mcrmse_loss = np.sqrt(mcrmse_loss)

predict = np.concatenate(valid['predict'], 0)
print(predict.shape)

submit_df = pd.DataFrame(data={
    'text_id': valid_df.text_id.values,
    'cohesion': predict[:, 0],  # check your dataset definition
    'syntax': predict[:, 1],
    'vocabulary': predict[:, 2],
    'phraseology': predict[:, 3],
    'grammar': predict[:, 4],
    'conventions': predict[:, 5],
})

score = do_local_evaluate(submit_df, valid_df)
print(score)
submit_df.to_csv('submission.csv', index=False)
print("success")

run_inference()