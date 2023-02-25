from dataset import *
from model import *
import os

def run_train():
    for fold in [0, 1, 2, 3]:

        out_dir = f'./result/run1/debertv3-base-meanpool-norm2-l1-02'
        fold_dir = f'{out_dir}/fold{fold}'
        if not os.path.exists(fold_dir):
            os.makedirs(fold_dir)


if __name__ == '__main__':
    run_train()
    print("success")