
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

# import spacy
# from spacy import displacy
import webbrowser


def read_text_and_clean(text_file):
	with open(text_file, 'r') as f:
		text = f.read()

	text = text.replace(u'\xa0', u' ')
	text = text.rstrip()
	text = text.lstrip()
	return text


#https://www.kaggle.com/code/kojimar/fb3-single-pytorch-model-train/notebook
def get_tokenizer():
	tokenizer = DebertaV2TokenizerFast.from_pretrained(arch)
	print('len(tokenizer)', len(tokenizer)) #128001
	return tokenizer


def make_fold(fold=0):
	num_fold = 4
	df = pd.read_csv(f'{root_dir}/data/feedback-prize-english-language-learning/train.csv')

	kf = MultilabelStratifiedKFold(n_splits=num_fold, shuffle=True, random_state=1234)
	for f, (t_index, v_index) in enumerate(kf.split(df, df[target_name])):
		if f==fold: break

	train_df = df.loc[t_index].reset_index(drop=True)
	valid_df = df.loc[v_index].reset_index(drop=True)
	return train_df, valid_df

# train_df, valid_df = make_fold(0)
# valid_df.to_csv(f'{root_dir}/data/other/valid_df.fold0.csv',index=False)
# exit(0)
# https://www.kaggle.com/cdeotte/pytorch-bigbird-ner-cv-0-615
# https://www.kaggle.com/cdeotte/tensorflow-longformer-ner-cv-0-633?scriptVersionId=83615733
'''
tokenizer.all_special_ids
Out[2]: [1, 2, 3, 0, 128000]
tokenizer.all_special_tokens
Out[3]: ['[CLS]', '[SEP]', '[UNK]', '[PAD]', '[MASK]']

'''


def text_to_token(text, tokenizer):

	encoded = tokenizer.encode_plus(
		text,
		add_special_tokens=True,
		return_offsets_mapping=False,
		max_length=None,#max_token_length,
		truncation=False, #True,
		#padding='max_length',
	) # return ['input_ids', 'attention_mask', 'offset_mapping']
	token_id   = encoded['input_ids']
	token_mask = encoded['attention_mask']
	#sum(token_mask)
	#------------------------------------------------------------------------
	#(np.array(token_id)!=0).sum()
	sample = {
		'token_id': token_id,
		'token_mask': token_mask,
	}
	return sample

##############################################################################################

class FeedbackDataset(Dataset):
	def __init__(self, df, tokenizer):
		self.df = df
		self.tokenizer = tokenizer
		self.length = len(self.df)

	def __str__(self):
		string = ''
		string += '\tlen = %d\n' % len(self)
		return string

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		d = self.df.iloc[index]
		sample = text_to_token(d.full_text, self.tokenizer)
		target = [
			d.cohesion,
			d.syntax,
			d.vocabulary,
			d.phraseology,
			d.grammar,
			d.conventions,
		]

		r = {}
		r['index'] = index
		r['d'] = d
		r['token_id'] = torch.LongTensor(sample['token_id'])
		r['token_mask'] = torch.FloatTensor(sample['token_mask'])
		r['target'] = torch.FloatTensor(target)
		return r


]




def null_collate(batch):
	#return d for model input
	d = {}
	#iterate the keys (i guess the output batch)
	key = batch[0].keys()
	for k in key:
		v = [b[k] for b in batch]
		if k in ['target']:
			v = torch.stack(v)
		d[k] = v
	#---
	L = [len(t) for t in d['token_id']]
	length = max(L)
	batch_size = len(d['token_id'])

	token_id = torch.full((batch_size,length),0) #PAD_id = 0
	token_mask = torch.full((batch_size,length),0)
	for b in range(batch_size):
		token_id[b,:L[b]]=d['token_id'][b]
		token_mask[b,:L[b]]=d['token_mask'][b]
	d['token_id'] = token_id
	d['token_mask'] = token_mask
	return d



##############################################################################################


def run_check_dataset():

	tokenizer = get_tokenizer()
	df = pd.read_csv(f'{root_dir}/data/feedback-prize-english-language-learning/train.csv')


	dataset = FeedbackDataset(df, tokenizer)
	print(dataset)
	for i in range(100):
		r = dataset[i]
		print(r['index'],'-----------')
		for k in tensor_list:
			v = r[k]
			print(k)
			print('\t', 'shape:', v.shape)
			print('\t', 'dtype:', v.dtype)
			print('\t', 'is_contiguous:', v.is_contiguous())
			print('\t', 'min/max:', v.min().item(), '/', v.max().item())
			print('\t', 'value:')
			print('\t\t', v.reshape(-1)[:8].tolist(), '...')
			print('\t\t', v.reshape(-1)[-8:].tolist())
		print('')

		if 1: #debug
			print('')
			token_id = r['token_id'].data.cpu().numpy().tolist()
			text = r['d'].full_text

			print('** token_id **')
			print(token_id)
			print('')

			print('** text **')
			print(text)
			print('')

 			#tokenizer.decode(token_id)
			print('** decode **')
			print(tokenizer.decode(token_id))
			print('')
			zz=0


	#----------
	loader = DataLoader(
		dataset,
		sampler = SequentialSampler(dataset),
		batch_size  = 4,
		drop_last   = True,
		num_workers = 0,
		pin_memory  = False,
		worker_init_fn = lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
		collate_fn = null_collate,
	)
	print( 'batch_size', loader.batch_size)
	print( 'len(loader)', len(loader))
	print( 'len(dataset)', len(dataset))
	print('')

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
			print('\t\t', v.reshape(-1)[:8])
		print('')

########################################################################################################################

if __name__ == "__main__":
	run_check_dataset()
	# start_lr = 1e-5
	# num_epoch =6.1
	#
	# x = [(start_lr - 1e-6) * (np.cos(epoch / num_epoch * np.pi) + 1) * 0.5 + 1e-6 for epoch in
	# 	 np.linspace(0, num_epoch, 30)]
	# plt.plot(x)
	# plt.show()