import re
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from sklearn.utils import shuffle
from config import Config

def PseudoLabeler(output_dir,model_name,sample_rate):
    pseudo = pd.read_csv(output_dir + '{}.csv'.format(model_name),header=None)
    data_test_ = pd.read_csv('/home/zhangruichang/sg2021/baseline/BERT/test_.tsv', header=None, sep='\t' )
    pseudo.columns=['label']
    data_test_.columns=['q1','q2']
    print(pseudo)
    print(data_test_)
    pseudo=pd.concat([pseudo,data_test_],axis=1,ignore_index = True)
    print(pseudo)
    num_of_samples = int(len(pseudo) * sample_rate)

    # Take a subset of the test set with pseudo-labels and append in onto
    # the training set
    pseudo_data = pseudo.sample(n=num_of_samples)
    train_data_path = 'train_.tsv'
    train_data = pd.read_csv(train_data_path, header=None, sep='\t')

    train_data.columns = ['label', 'q1', 'q2']
    pseudo_data.columns = ['label', 'q1', 'q2']

    r = "[_.!+-=——,$%^，。？?、~@#￥%……&*《》<>「」{}【】()/\\\[\]]"
    for i in trange(train_data.shape[0]):  # tqdm(range(x)):用于显示进度条
        train_data['q1'].iloc[i] = re.sub(r, '', train_data['q1'].iloc[i])
        train_data['q2'].iloc[i] = re.sub(r, '', train_data['q2'].iloc[i])

    for i in trange(pseudo_data.shape[0]):  # tqdm(range(x)):用于显示进度条
        pseudo_data['q1'].iloc[i] = re.sub(r, '', pseudo_data['q1'].iloc[i])
        pseudo_data['q2'].iloc[i] = re.sub(r, '', pseudo_data['q2'].iloc[i])

    augemented_train = pd.concat([train_data,pseudo_data],axis=0,ignore_index = True)

    return shuffle(augemented_train)

if __name__ == '__main__':
    config = Config()
    output = '/home/zhangruichang/sg2021/baseline/BERT/pseudo_train_data/'
    output_dir = config.infer_output_dir
    model_name = config.infer_model_name
    sample_rate = 1.0
    augemented_train = PseudoLabeler(output_dir,model_name,sample_rate)
    augemented_train.to_csv(output + 'pseudo_train_{}.tsv'.format(model_name),index=None)

