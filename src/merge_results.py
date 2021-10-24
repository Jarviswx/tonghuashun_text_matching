import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from tqdm import tqdm, trange
import json
import torch
import os

num_of_models = 3
y_pred_list=[]
#data1 = pd.read_csv('pseudo_results/0.5853_Pred_train_robert_we_bertlstm_msdrop_fl_lr.csv',header=None).iloc[:,0].to_list()
#data2 = pd.read_csv('pseudo_results/0.587_Pred_noargumented_bert_best_macbert_tcnn_msdrop_lr_fl.csv',header=None).iloc[:,0].to_list()
data3 = pd.read_csv('pseudo_results/0.5889_Pred_bert_best_macbert_sbert_coatten_fgm_fl_lr.csv',header=None).iloc[:,0].to_list()
data4 = pd.read_csv('results/0.5712_Pred_noargumented_bert_best_nezha_bw_nezha_fgm_fl_lr.csv',header=None).iloc[:,0].to_list()
data5 = pd.read_csv('results/0.574_y_Pred_bert_best_ernie_tcnn_msdrop_fl_lr.csv',header=None).iloc[:,0].to_list()

output_dir = '/home/zhangruichang/sg2021/baseline/BERT/results/'

#y_pred_list.append(data1)
#y_pred_list.append(data2)
y_pred_list.append(data3)
y_pred_list.append(data4)
y_pred_list.append(data5)

y_pred_array = np.array(y_pred_list)
y_pred_array = y_pred_array.T
y_pred_final = []
for i in trange(y_pred_array.shape[0]):
    count = np.bincount(y_pred_array[i])
    label = np.argmax(count)
    y_pred_final.append(label)


with open(output_dir+'ee0.5853+0.587+0.5889_merge_{}.json'.format(num_of_models), 'w+') as f:
    #     labels_list = [{'label':str(label)} for label in list(label_predict_list)]
    for label in list(y_pred_final):
        f.write(json.dumps({'label': int(label)}) + '\n')

data_3 =pd.DataFrame(y_pred_final)
data_3.to_csv(output_dir+'ee0.5853+0.587+0.5889_merge_{}.csv'.format(num_of_models),index=None,header=None)
