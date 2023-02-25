from transformers import AutoTokenizer,AutoConfig,AutoModel, DebertaV2Model
import torch
import torch.nn as nn
import torch.nn.functional as F

model_name = "microsoft/deberta-v3-small"

class Net(nn.Module):
    def __init__(self,num_classes,model_name = "microsoft/deberta-v3-small"):
        super(Net,self).__init__()
        #self.output_type =  output_type
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.predict = nn.Linear(self.model_config.hidden_size, num_classes)

    def forward(self,batch):
        token_id = batch['input_ids']
        token_mask = batch['attention_mask']
        print("model_token_id size = ", token_id.shape)
        print("model_token_mask size = ", token_mask.shape)
        out = self.transformer(token_id, token_mask)
        x = out.last_hidden_state[:,0]#??
        logit = self.predict(x)
        return logit

if __name__ == "__main__":
    num_classes = 5
    net = Net(num_classes)
    text = ["This is an example sentence to tokenize.",
            "another sentence",
            "one more"]
    tokenizers = AutoTokenizer.from_pretrained(model_name)
    input = tokenizers(text,padding=True,truncation=True,return_tensors="pt")
    print(type(input['input_ids']),input['input_ids'].shape)
    print(type(input['attention_mask']),input['attention_mask'].shape)
    y = net(input)
    print(y)

