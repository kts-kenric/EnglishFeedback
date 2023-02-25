
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


num_target = 6
num_sample = 100
text_id = np.random.choice(1000,num_sample)

truth = np.ones((num_sample,num_target))
predict = truth
predict = np.where(np.random.random(predict.shape) < 0.5, 0, predict)

truth_df = pd.DataFrame(data={
        'text_id'     : text_id,
        'cohesion'    : truth[:,0], #check your dataset definition
        'syntax'      : truth[:,1] ,
        'vocabulary'  : truth[:,2] ,
        'phraseology' : truth[:,3] ,
        'grammar'     : truth[:,4] ,
        'conventions' : truth[:,5] ,
    })

predict_df = pd.DataFrame(data={
        'text_id'     : text_id,
        'cohesion'    : predict[:,0], #check your dataset definition
        'syntax'      : predict[:,1] ,
        'vocabulary'  : predict[:,2] ,
        'phraseology' : predict[:,3] ,
        'grammar'     : predict[:,4] ,
        'conventions' : predict[:,5] ,
    })



score = do_local_evaluate(predict_df, truth_df)
print(score)