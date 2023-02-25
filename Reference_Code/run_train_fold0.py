import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from common import *
from my_lib.net.lookahead import *

import torch.cuda.amp as amp
is_amp = True  #True #False

from dataset import *
from model import *


#https://github.com/rssalessio/PytorchRBFLayer
##########################################################################################################3
def do_valid(net, valid_loader):

	valid_metric = 0
	valid_loss = 0
	valid_num  = 0

	net.eval()
	net.output_type = ['inference', 'loss']
	start_timer = timer()
	for t, batch in enumerate(valid_loader):
		batch_size = len(batch['index'])
		for k in tensor_list: batch[k] = batch[k].cuda()

		with torch.no_grad():
			with amp.autocast(enabled=is_amp):

				output = data_parallel(net,batch)
				loss0  = output['l1_loss'].mean()
				#loss0  = output['mse_loss'].mean()
				loss1  = output['mcrmse_loss']#.mean()


		valid_num += batch_size
		valid_loss += batch_size*loss0.item()
		valid_metric += batch_size*np.square(loss1.data.cpu().numpy())
		# ------------------------------------------------
		print('\r %8d / %d  %s'%(valid_num, len(valid_loader.dataset),time_to_str(timer() - start_timer,'sec')),end='',flush=True)

	torch.cuda.empty_cache()
	assert(valid_num == len(valid_loader.dataset))
	#print('')
	#----------------------
	mse_loss = valid_loss/valid_num
	mcrmse_loss = valid_metric/valid_num
	mcrmse_loss = np.sqrt(mcrmse_loss)

	return [mse_loss, mcrmse_loss.mean(), 0]



##########################################################################################################
# https://github.com/jamescalam/transformers/blob/main/course/training/03_mlm_training.ipynb
def run_train_test(

):
	fold = 0
	#if 1:
	for fold in [0,1,2,3]:
		out_dir = f'{root_dir}/result/run51/debertv3-base-meanpool-norm2-l1-02'
		fold_dir = f'{out_dir}/fold{fold}'
		initial_checkpoint = \
		    None #fold_dir + '/checkpoint/00004168.model.pth'  #

		# 1e-2 4: 1e-4  10
		num_epoch     = 10.1
		batch_size    = 4
		skip_epoch_save = 3

		start_lr = 1e-3
		min_lr = 1e-6
		cycle  = 6

		## setup  ----------------------------------------
		for f in ['checkpoint', 'train', 'valid', 'backup']: os.makedirs(fold_dir + '/' + f, exist_ok=True)
		# backup_project_as_zip(PROJECT_PATH, out_dir +'/backup/code.train.%s.zip'%IDENTIFIER)

		log = Logger()
		log.open(fold_dir + '/log.train.txt', mode='a')
		log.write(f'\n--- [START {log.timestamp()}] {"-" * 64}\n\n')
		log.write(f'\t{set_environment()}\n')
		log.write(f'\t__file__ = {__file__}\n')
		log.write(f'\tfold_dir = {fold_dir}\n')
		log.write('\n')

		## dataset ------------------------------------
		tokenizer = get_tokenizer()
		train_df, valid_df = make_fold(fold)
		train_dataset = FeedbackDataset(train_df, tokenizer)
		valid_dataset = FeedbackDataset(valid_df, tokenizer)

		#---

		train_loader = DataLoader(
			train_dataset,
			sampler = RandomSampler(train_dataset),
			batch_size  = batch_size,
			drop_last   = True,
			num_workers = 0,
			pin_memory  = False,
			worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
			collate_fn = null_collate,
		)
		valid_loader  = DataLoader(
			valid_dataset,
			sampler = SequentialSampler(valid_dataset),
			batch_size  = 8,
			drop_last   = False,
			num_workers = 0,
			pin_memory  = False,
			collate_fn = null_collate,
		)

		log.write('fold  : %d\n'%(fold))
		log.write('train_dataset : \n%s\n'%(train_dataset))
		log.write('valid_dataset : \n%s\n'%(valid_dataset))
		log.write('\n')

		#train_dataset = valid_dataset#

		## net ----------------------------------------
		log.write('** net setting **\n')
		scaler = amp.GradScaler(enabled = is_amp)
		net = Net().cuda()


		if initial_checkpoint is not None:
			f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
			start_iteration = f['iteration']
			start_epoch = f['epoch']
			state_dict = f['state_dict']
			net.predict.load_state_dict(state_dict,strict=False)  #True

		else:
			start_iteration = 0
			start_epoch = 0


		log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
		log.write('\n')

		# -----------------------------------------------
		if 1: ##freeze
			for p in net.transformer.parameters():
				p.requires_grad = False

			# for i in range(0):
			# 	L = len(net.transformer.encoder.layer)
			# 	for p in net.transformer.encoder.layer[L-1-i].parameters():
			# 		p.requires_grad = True

			# for i in range(1):
			# 	L = len(net.transformer.encoder.layer)
			# 	for p in net.transformer.encoder.layer[i].parameters():
			# 		p.requires_grad = True
			#
			# for p in net.transformer.embeddings.parameters():
			# 	p.requires_grad = False

			#for p in net.backbone.backbone.stem.parameters(): p.requires_grad = True

		#------------------------------------
		#param = net.parameters()  #list(seed_net.parameters()) + list(color_net.parameters()) #seed_net.parameters() #
		optimizer = Lookahead(RAdam(filter(lambda p: p.requires_grad, net.parameters() ), lr=start_lr), alpha=0.5, k=5)
		#optimizer = RAdam(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr)
		#optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr, momentum=0.9)
		#optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),lr=start_lr,)

		'''
		
		 def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
		
		'''

		# transformer_parameter = list(net.transformer.named_parameters())
		# other_parameter = list(net.predict.named_parameters())
		# no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
		# parameter = [
		# 	{
		# 		'params': [p for n, p in transformer_parameter if not any(nd in n for nd in no_decay)],
		# 		'weight_decay': 0.01,
		# 	},
		# 	{
		# 		'params': [p for n, p in transformer_parameter if any(nd in n for nd in no_decay)],
		# 		'weight_decay': 0.0,
		# 	},
		# 	{
		# 		'params': [p for n, p in other_parameter],
		# 		'weight_decay': 0.0,
		# 	},
		# ]
		# optimizer = optim.AdamW(parameter, lr=start_lr)
		# #------------------------------------


		num_iteration = num_epoch * len(train_loader)
		iter_log = int(len(train_loader) * 1)  #
		iter_valid = iter_log
		iter_save  = iter_log

		log.write('optimizer\n  %s\n'%(optimizer))
		log.write('\n')


		## start training here! ##############################################
		log.write('** start training here! **\n')
		log.write('   batch_size = %d \n' % (batch_size))
		log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
		log.write('                     |-------------- VALID---------|---- TRAIN/BATCH ----------------\n')
		log.write('rate     iter  epoch | dice   loss   tp     tn     | loss           | time           \n')
		log.write('-------------------------------------------------------------------------------------\n')

		# 0.00100   0.50  0.80 | 0.891  0.020  0.000  0.000  | 0.000  0.000   |  0 hr 02 min

		def message(mode='print'):
			asterisk = ' '
			if mode == ('print'):
				loss = batch_loss
			if mode == ('log'):
				loss = train_loss
				if (iteration % iter_save == 0): asterisk = '*'

			text = \
				('%0.2e   %08d%s %6.2f | ' % (rate, iteration, asterisk, epoch,)).replace('e-0', 'e-').replace('e+0',
																											   'e+') + \
				'%4.3f  %4.3f  %4.4f  | ' % (*valid_loss,) + \
				'%4.3f  %4.3f  %4.3f  | ' % (*loss,) + \
				'%s' % (time_to_str(timer() - start_timer, 'min'))

			return text

		#----
		valid_loss = np.zeros(3,np.float32)
		train_loss = np.zeros(3,np.float32)
		batch_loss = np.zeros_like(train_loss)
		sum_train_loss = np.zeros_like(train_loss)
		sum_train = 0


		start_timer = timer()
		iteration = start_iteration
		epoch = start_epoch
		rate = 0
		while iteration < num_iteration:

			# if epoch >= 3:
			# start_lr *= 0.9
			# lr = start_lr*np.pow(0.9,epoch)
			#lr =  (start_lr - min_lr) * (np.cos(epoch / num_epoch * np.pi) + 1) * 0.5 + min_lr
			#adjust_learning_rate(optimizer, lr)
			# lr = scheduler(epoch)
			# optimizer = optim.AdamW(set_layerwise_learning_rate(net, lr), lr=lr)


			for t, batch in enumerate(train_loader):

				#-----------------
				if iteration%iter_save==0:
					if iteration != start_iteration:
						if epoch<skip_epoch_save:
							n = 0
						else:
							n = iteration
						torch.save({
							'state_dict': net.predict.state_dict(),
							'iteration': iteration,
							'epoch': epoch,
						}, fold_dir + '/checkpoint/%08d.model.pth' % n)
						pass
				#if 1:
				if (iteration % iter_valid == 0):
					#if iteration!=start_iteration:
						valid_loss = do_valid(net, valid_loader)  #
						pass

				if (iteration % iter_log == 0):
					print('\r', end='', flush=True)
					log.write(message(mode='log') + '\n')


				# learning rate schduler ------------
				if epoch<cycle:
					lr = (start_lr - min_lr) * (np.cos(epoch / cycle * np.pi) + 1) * 0.5 + min_lr
				else:
					lr = min_lr
				adjust_learning_rate(optimizer, lr)

				rate = get_learning_rate(optimizer)[0] #scheduler.get_last_lr()[0] #get_learning_rate(optimizer)

				# one iteration update  -------------
				batch_size = len(batch['index'])
				for k in tensor_list: batch[k] = batch[k].cuda()


				net.train()
				net.output_type = ['loss']
				optimizer.zero_grad()
				with amp.autocast(enabled = is_amp):
					output = data_parallel(net,batch)
					loss0  = output['l1_loss'].mean()
					#loss0 = output['mse_loss'].mean()
					loss1  = output['mcrmse_loss'].mean()

					scaler.scale(loss0).backward()
					scaler.unscale_(optimizer)
					#torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
					scaler.step(optimizer)
					scaler.update()

				# print statistics  --------
				epoch += 1 / len(train_loader)
				iteration += 1
				if iteration > num_iteration: break

				batch_loss = np.array([loss0.item(), loss1.item(), 0])
				sum_train_loss += batch_loss
				sum_train += 1
				if iteration % min(iter_log,100) == 0:
					train_loss = sum_train_loss / (sum_train + 1e-12)
					sum_train_loss[...] = 0
					sum_train = 0

				print('\r', end='', flush=True)
				print(message(mode='print'), end='', flush=True)


				# debug------------------------------

				# if is_debug!=0:
				# 	pass

		torch.cuda.empty_cache()
		log.write('\n')


# main #################################################################
if __name__ == '__main__':
	run_train()
