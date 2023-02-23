import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoConfig,AutoModel, DebertaV2Model

class Net(nn.module):
    def __init__(self,model_name = "microsoft/deberta-v3-small"):
        super().__init__()
        self.model_config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)

        #what i need config file,
        #model pretrained

