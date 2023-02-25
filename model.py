import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoConfig,AutoModel, DebertaV2Model
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES']='0'

class Net(nn.Module):
    def __init__(self,num_classes,model_name = "microsoft/deberta-v3-small"):
        super().__init__()
        self.output_type = ['inference', 'loss']
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.predict = nn.Sequential(
            nn.Linear(self.model_config.hidden_size, 64),
            nn.Linear(64, num_classes),

        )
        #what i need config file,
        #model pretrained

    def forward(self,batch):
        token_id = batch['token_id']
        token_mask = batch['token_mask']
        model_out = self.transformer(token_id,token_mask)
        x = model_out.last_hidden_state[:, 0]
        out = self.predict(x)
        output = {}
        if 'loss' in self.output_type:
            output['mse_loss'] = F.mse_loss(out,batch['target'])
            output['l1_loss'] = F.smooth_l1_loss(out,batch['target'])

        if 'inference' in self.output_type:
            output['predict'] = out
        return output


def run_check_net():
    batch_size = 1
    vocab_size = 5000
    max_sequence_length = 512
    num_classes = 5

    #debug Test

    #initalize all the values for testing
    #remember that your input you need token_id, token_mask and target
    token_id = np.random.choice(vocab_size,(batch_size,max_sequence_length))
    token_mask = np.random.choice(2,(batch_size,max_sequence_length))
    token_id = torch.from_numpy(token_id).long()
    token_mask = torch.from_numpy(token_mask).float()
    target = np.random.uniform(0,5,(batch_size, num_classes))
    target = torch.from_numpy(target).float()

    batch = {
        'token_id':token_id.cuda(),
        'token_mask':token_mask.cuda(),
        'target':target.cuda()
    }

    net = Net(num_classes).cuda()

    #how to get predict batch (remember its a dict )
    out = net(batch)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            out = net(batch)

    print('batch')
    for k, v in batch.items():
        print('%32s :' % k, v.shape)

    print('output')
    for k, v in out.items():
        if 'loss' not in k:
            print('%32s :' % k, v.shape)
    for k, v in out.items():
        if 'loss' in k:
            if len(v.shape)==0:
                print('%32s :' % k, v.item())
            else:
                print('%32s :' % k, v.data.cpu().numpy().tolist())
    print("zzz")
if __name__ == '__main__':
    run_check_net()