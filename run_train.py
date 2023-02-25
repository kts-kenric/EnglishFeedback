from dataset import *
from model import *
import os
from Lib.lookahead import *


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

ds_train_path = "train.csv"
ds_test_path = "test.csv"
num_classes = 5


def run_train():
    for fold in [0, 1, 2, 3]:

        out_dir = f'./result/run1/debertv3-base-meanpool-norm2-l1-02'
        fold_dir = f'{out_dir}/fold{fold}'
        initial_checkpoint = None #fold_dir + '/checkpoint/00004168.model.pth'

        num_epoch = 10
        batch_size = 2
        skip_epoch_save = 3

        #start learning rate
        start_lr = 1e-3
        min_lr = 1e-6
        #??
        cycle = 6
        #setup file_path
        for f in ['checkpoint', 'train', 'valid', 'backup']: os.makedirs(fold_dir + '/' + f, exist_ok=True)

        print(f'\n--- [START] ---\n\n')
        print(f'\t__file__ = {__file__}\n')
        print(f'\tfold_dir = {fold_dir}\n')
        print('print\n')

        train_df, valid_df = make_fold(ds_train_path, ds_test_path)
        tokenizer = get_tokenizer()
        train_dataset = EnglishDataset(train_df, tokenizer)
        valid_dataset = EnglishDataset(valid_df, tokenizer)

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,
            num_workers=0,
            pin_memory=False,  # speed up the host to device transfer usually
            sampler=SequentialSampler(train_dataset), #can train RandomSampler as well
            # seed for worker process so each process generates a unique seed
            worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
            collate_fn=null_collate,
            drop_last=False,

        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=4,
            num_workers=0,
            pin_memory=False,  # speed up the host to device transfer usually
            sampler=SequentialSampler(valid_dataset),
            collate_fn=null_collate,
            drop_last=False,
        )

        print('fold  : %d\n' % (fold))
        print('train_dataset : \n%s\n' % (train_dataset))
        print('valid_dataset : \n%s\n' % (valid_dataset))
        print('\n')

        #scaler = amp.GradScaler(enabled=is_amp)

        net = Net(num_classes).cuda()

        if initial_checkpoint is not None:
            #load the inital check point
            f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
            start_iteration = f['iteration']
            start_epoch = f['epoch']
            state_dict = f['state_dict']
            net.predict.load_state_dict(state_dict, strict=False)

        else:
            start_iteration = 0
            start_epoch = 0

        print.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        print.write('\n')

        if 1:
            #iterate through all the transformer layer define in the model and freeze them
            #will only train the predict layer
            #this is so that we can leverge knowledged that is learnt by transformer layer
            #for a task with a smaller dataset
            for p in net.transformer.parameters():
                p.requires_grad = False




        optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters()), lr=start_lr), alpha=0.5, k=5)

if __name__ == '__main__':
    run_train()
    print("success")