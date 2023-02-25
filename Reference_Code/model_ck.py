import torch
from common import *
from configure import *


#https://www.kaggle.com/code/kojimar/fb3-single-pytorch-model-train
def re_initialize(net,):
    L = len(net.transformer.encoder.layer)
    i = L-1
    for module in net.transformer.encoder.layer[i].modules():
        #print(module)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    return net



def mean_pool(x, attention_mask):
    m = attention_mask.unsqueeze(-1).expand(x.size()).float()
    sum_embedding = torch.sum(x * m, 1)
    sum_mask = torch.sum(m, 1)
    sum_mask = torch.clamp(sum_mask, min=1e-9)
    mean = sum_embedding / sum_mask
    return mean

#----------------------------------
class Net(nn.Module):
    def __init__(self,):
        super().__init__()
        self.output_type = ['inference', 'loss']


        config = AutoConfig.from_pretrained(arch)
        config.update(
            {
                'output_hidden_states': True,
                #'hidden_dropout_prob': 0.,
                'hidden_dropout' : 0.,
                'hidden_dropout_prob' : 0.,
                'attention_dropout' : 0.,
                'attention_probs_dropout_prob': 0.,
                'layer_norm_eps':  1e-7,
                'add_pooling_layer': False,
                'num_labels': 1,
            }
        )
        self.transformer = AutoModel.from_pretrained(arch, config=config)
        ###self.transformer.resize_token_embeddings(128101) #len(tokenizer))
        ###self.transformer.encoder.LayerNorm=None

        # self.L = len(self.transformer.encoder.layer)
        # self.aux = nn.ModuleList([
        #     nn.Linear(config.hidden_size, num_target) for i in range(3)
        # ])


        self.predict = nn.Sequential(
            nn.Linear(config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.SiLU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),

            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.SiLU(),

            nn.Linear(512, num_target),
        )



    def forward(self, batch):

        token_id = batch['token_id']
        token_mask = batch['token_mask']
        out = self.transformer(token_id, token_mask)

        #x = torch.cat([out.hidden_states[self.L-1-i][:,0] for i in range(3)],-1)

        #x = out.last_hidden_state[:,0]
        #out = sum(out.hidden_states)
        x = mean_pool(out.last_hidden_state, token_mask)
        #x = F.normalize(x,p=2,dim=1)
        #x = out.last_hidden_state.mean(1)

        predict = self.predict(x)
        # aux = [
        #     #self.aux[i](out.hidden_states[i][:,0]) for i in range(self.L)
        #     self.aux[i](
        #         mean_pool(out.hidden_states[self.L-1-i], token_mask)
        #     ) for i in range(3)
        # ]

        output = {}
        if 'loss' in self.output_type:
            # output['mse_loss'] = F.mse_loss(predict,batch['target'])
            output['l1_loss'] = F.smooth_l1_loss(predict,batch['target'])
            output['mcrmse_loss'] = criterion_mcrmse(predict,batch['target'])
            #output['aux_loss'] = criterion_aux(aux,batch['target'])

        if 'inference' in self.output_type:
            output['predict'] = predict
        return output


def criterion_mcrmse(predict, truth):
    diff = (predict-truth)**2
    loss = diff.mean(0)
    loss = torch.sqrt(loss)
    return loss

def criterion_aux(predict, truth):
    loss = sum([F.smooth_l1_loss(p,truth) for p in predict])
    return loss



def run_check_net():
    batch_size = 16
    max_length = 512
    vocab_size = 5000


    token_id = np.random.choice(vocab_size,(batch_size, max_length))
    token_id = torch.from_numpy(token_id).long()

    token_mask = np.random.choice(2,(batch_size, max_length))
    token_mask = torch.from_numpy(token_mask).float()


    target = np.random.uniform(0,5,(batch_size, num_target))
    target = torch.from_numpy(target).float()

    batch = {
        'token_id' : token_id.cuda(),
        'token_mask' : token_mask.cuda(),
        'target' : target.cuda(),
    }

    #----

    net = Net().cuda()
    #print(net)
    output = net(batch)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=True):
            output = net(batch)

    print('batch')
    for k, v in batch.items():
        print('%32s :' % k, v.shape)

    print('output')
    for k, v in output.items():
        if 'loss' not in k:
            print('%32s :' % k, v.shape)
    for k, v in output.items():
        if 'loss' in k:
            if len(v.shape)==0:
                print('%32s :' % k, v.item())
            else:
                print('%32s :' % k, v.data.cpu().numpy().tolist())


# main #################################################################
if __name__ == '__main__':
    run_check_net()



