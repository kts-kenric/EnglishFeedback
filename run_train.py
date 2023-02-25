from dataset import *
from model import *
import os
from Lib.lookahead import *
from Helper.print import *
from Lib.rate import *
import torch.cuda.amp as amp

#config
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
is_amp = True  #True #False

ds_train_path = "train.csv"
ds_test_path = "test.csv"
num_classes = 6

def do_valid(net, valid_loader):
    valid_metric = 0
    valid_loss = 0
    valid_num  = 0

    net.eval()
    net.output_type = ['inference', 'loss']
    start_timer = time.time()
    for t, batch in enumerate(valid_loader):
        batch_size = len(batch['index'])
        for k in tensor_list: batch[k] = batch[k].cuda()

        with torch.no_grad():
            with amp.autocast(enabled=is_amp):

                output = torch.nn.parallel.data_parallel(net,batch)
                loss0  = output['l1_loss'].mean()
                #loss0  = output['mse_loss'].mean()
                loss1  = output['mcrmse_loss']#.mean()


        valid_num += batch_size
        valid_loss += batch_size*loss0.item()
        valid_metric += batch_size*np.square(loss1.data.cpu().numpy())

        # ------------------------------------------------
        print('\r %8d / %d %s %s'%(valid_num, len(valid_loader.dataset),str(time.time() - start_timer),'sec'),end='',flush=True)

    torch.cuda.empty_cache()
    assert(valid_num == len(valid_loader.dataset))
    #print('')
    #----------------------
    mse_loss = valid_loss/valid_num
    mcrmse_loss = valid_metric/valid_num
    mcrmse_loss = np.sqrt(mcrmse_loss)

    return [mse_loss, mcrmse_loss.mean(), 0]
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

        scaler = amp.GradScaler(enabled=is_amp)

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

        print('\tinitial_checkpoint = %s\n' % initial_checkpoint)
        print('\n')

        if 1:
            #iterate through all the transformer layer define in the model and freeze them
            #will only train the predict layer
            #this is so that we can leverge knowledged that is learnt by transformer layer
            #for a task with a smaller dataset
            for p in net.transformer.parameters():
                p.requires_grad = False


        #ref
        #https://lessw.medium.com/new-deep-learning-optimizer-ranger-synergistic-combination-of-radam-lookahead-for-the-best-of-2dc83f79a48d

        #lambda p: p.requires_grad, net.parameters()
        #this line takes net.parameters and will return when it is true
        #filter will filter to those that is true only
        base_optim = RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr = start_lr)
        optimizer = Lookahead(base_optim, k=5, alpha=0.5)

        #total number of iteration
        num_iteration = num_epoch * len(train_loader)

        iter_log = int(len(train_loader) * 1)  #
        iter_valid = iter_log
        iter_save = iter_log

        print('optimizer\n  %s\n' % (optimizer))
        print('\n')

        ## start training here! ##############################################
        print('** start training here! **\n')
        print('   batch_size = %d \n' % (batch_size))
        print('   experiment = %s\n' % str(__file__.split('/')[-2:]))
        print('                     |-------------- VALID---------|---- TRAIN/BATCH ----------------\n')
        print('rate     iter  epoch | dice   loss   tp     tn     | loss           | time           \n')
        print('-------------------------------------------------------------------------------------\n')


        valid_loss = np.zeros(3, np.float32)
        train_loss = np.zeros(3, np.float32)
        batch_loss = np.zeros(3, np.float32)
        sum_train_loss = np.zeros_like(train_loss)
        sum_train = 0

        start_timer = time.time()
        iteration = start_iteration
        epoch = start_epoch
        rate = 0

        while iteration < num_iteration:
            #for the dataset, len(train_loader) will be based on the the batch amount set
            for t, batch in enumerate(train_loader):
                # -----------------
                if iteration % iter_save == 0:
                    if iteration != start_iteration:
                        if epoch < skip_epoch_save:
                            n = 0
                        else:
                            n = iteration
                        torch.save({
                            'state_dict': net.predict.state_dict(),
                            'iteration': iteration,
                            'epoch': epoch,
                        }, fold_dir + '/checkpoint/%08d.model.pth' % n)
                        pass
                # if 1:
                if (iteration % iter_valid == 0):
                    # if iteration!=start_iteration:
                    valid_loss = do_valid(net, valid_loader)  #
                    pass

                if (iteration % iter_log == 0):
                    print('\r', end='', flush=True)
                    #print(message(mode='log') + '\n')
                    print(message(batch_loss, train_loss, valid_loss, iteration, iter_save, rate, epoch, start_timer,
                                  mode='log'), + '\n')

                # learning rate schduler ------------
                if epoch < cycle:
                    lr = (start_lr - min_lr) * (np.cos(epoch / cycle * np.pi) + 1) * 0.5 + min_lr
                else:
                    lr = min_lr
                adjust_learning_rate(optimizer, lr)

                rate = get_learning_rate(optimizer)[0]  # scheduler.get_last_lr()[0] #get_learning_rate(optimizer)

                # one iteration update  -------------
                batch_size = len(batch['index'])
                for k in tensor_list: batch[k] = batch[k].cuda()

                net.train()
                net.output_type = ['loss']
                optimizer.zero_grad()
                #with amp.autocast(enabled=is_amp):
                output = torch.nn.parallel.data_parallel(net, batch)
                loss0 = output['l1_loss'].mean()
                # loss0 = output['mse_loss'].mean()
                loss1 = output['mcrmse_loss'].mean()

                scaler.scale(loss0).backward()
                scaler.unscale_(optimizer)
                # torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                scaler.step(optimizer)
                scaler.update()

                # print statistics  --------
                epoch += 1 / len(train_loader)
                iteration += 1
                if iteration > num_iteration: break

                batch_loss = np.array([loss0.item(), loss1.item(), 0])
                sum_train_loss += batch_loss
                sum_train += 1
                if iteration % min(iter_log, 100) == 0:
                    train_loss = sum_train_loss / (sum_train + 1e-12)
                    sum_train_loss[...] = 0
                    sum_train = 0

                print('\r', end='', flush=True)

                print(message(batch_loss, train_loss, valid_loss, iteration, iter_save, rate, epoch, start_timer,
                            mode='print'), end='', flush=True)

                #print(message(mode='print'), end='', flush=True)

                # debug------------------------------

                # if is_debug!=0:
                # 	pass

            torch.cuda.empty_cache()
            print('\n')


if __name__ == '__main__':
    run_train()
    print("success")