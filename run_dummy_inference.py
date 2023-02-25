from dataset import *
from model import *
import os
from Lib.lookahead import *
from Helper.print import *
from Lib.rate import *
import torch.cuda.amp as amp
is_amp = True  #True #False
ds_train_path = "train.csv"
ds_test_path = "test.csv"
num_classes = 6

def do_local_evaluate(submit_df, truth_df):
    assert(submit_df['text_id'].equals(truth_df['text_id']))
    predict = submit_df[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']].values
    truth = truth_df[['cohesion','syntax','vocabulary','phraseology','grammar','conventions']].values
    d = ((truth - predict)**2)
    d = d.mean(axis=0)
    d = d**0.5
    score = d.mean(axis=0)
    print("zzz")
    ## ..compute here
    return score


def run_inference():
    initial_checkpoint = "./result/run1/debertv3-base-meanpool-norm2-l1-02/fold0/checkpoint/00001764.model.pth"
    train_df, valid_df = make_fold(ds_train_path, ds_test_path)
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

    submit_df = pd.DataFrame(data={
        'text_id'     : valid_df.text_id.values,
        'cohesion'    : 3 , #check your dataset definition
        'syntax'      : 3 ,
        'vocabulary'  : 3 ,
        'phraseology' : 3 ,
        'grammar'     : 3 ,
        'conventions' : 3 ,
    })

    score = do_local_evaluate(submit_df, valid_df)
    print(score)
    submit_df.to_csv('submission.csv', index=False)
'''
1st run
3128 / 3128   5 min 26 sec
(3128, 6)
zzz
0.6216551958132578
'''
if __name__ == '__main__':
    run_inference()

