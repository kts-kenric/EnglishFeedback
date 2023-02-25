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
    valid_num  = 0

    net.cuda()
    net.eval()
    net.output_type = ['inference', 'loss']
    start_timer = time.time()
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

        # ------------------------------------------------
        print('\r %8d / %d  %s' % (valid_num, len(valid_loader.dataset), time_to_str(time.time() - start_timer, 'sec')),
              end='', flush=True)

    print('')
    assert (valid_num == len(valid_loader.dataset))

#----------------------
    mse_loss    = valid['loss']/valid_num
    mcrmse_loss = valid['metric']/valid_num
    mcrmse_loss = np.sqrt(mcrmse_loss)

    predict = np.concatenate(valid['predict'],0)
    print(predict.shape)

    submit_df = pd.DataFrame(data={
        'text_id'     : valid_df.text_id.values,
        'cohesion'    : predict[:,0], #check your dataset definition
        'syntax'      : predict[:,1] ,
        'vocabulary'  : predict[:,2] ,
        'phraseology' : predict[:,3] ,
        'grammar'     : predict[:,4] ,
        'conventions' : predict[:,5] ,
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

'''
private 0.60
public 0.58
'''

if __name__ == '__main__':
    run_inference()

