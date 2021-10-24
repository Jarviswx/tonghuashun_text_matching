#导入所需第三方库
import os
import re
import json
import numpy as np
import pandas as pd
from random import random
from tqdm import tqdm, trange
from collections.abc import Iterable
from sklearn.metrics import f1_score
from sklearn.model_selection import KFold
from transformers import BertModel, BertConfig, BertTokenizer, AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_with_hard_restarts_schedule_with_warmup
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
#导入所需文件
from config import Config
from model import BertClassifierSingleModel, NezhaClassifierSingleModel, SBERTSingleModel, SNEZHASingleModel_MDP, BertClassifierTextCNNSingleModel_MDP, SBERTCoAttentionModel_MDP, SBERTCoAttentionModel,  BERTwithMDP, BertLstm_MDP, SNEZHASingleModel
from utils import focal_loss, FGM, RAdam, Lookahead
#指定使用的GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def infer(device,num_of_models,pretrained,hidden_size,saved_model_path,infer_model_name,output_filename,output_dir):

    # 处理数据
    class TestData(Dataset):
        def __init__(self, train_data, max_length, tokenizer):
            self.data = train_data
            self.max_length = max_length
            self.max_num_tokens = max_length - 3

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            q1 = self.data['q1'].iloc[idx]
            q2 = self.data['q2'].iloc[idx]
            tokens_a = tokenizer.tokenize(q1)
            tokens_b = tokenizer.tokenize(q2)
            truncate_seq_pair(tokens_a, tokens_b, self.max_num_tokens)
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
            assert len(segment_ids) == len(input_ids)

            padding_length = self.max_length - len(input_ids)
            input_ids += [0] * padding_length
            segment_ids += [0] * padding_length
            assert len(segment_ids) == len(input_ids) == self.max_length
            input_ids = torch.tensor(input_ids, dtype=torch.long)
            segment_ids = torch.tensor(segment_ids, dtype=torch.long)
            return input_ids, segment_ids

    # 选择自定义模型（Bert类和Nezha）
    print('======================自定义模型,加载预训练模型的config======================')
    print(infer_model_name)
    if 'bert' in infer_model_name or 'sbert' in infer_model_name:
        print('使用Bert模型')
        if 'cnn' in infer_model_name:
            print('模型名称：bert_textcnn')
            if 'msdrop' in infer_model_name:
                print('使用multi_sampel_drop')
                model = BertClassifierTextCNNSingleModel_MDP(bert_dir=pretrained,from_pretrained=False,hidden_size=hidden_size)
            else:
                print('不使用multi_sample_drop')
                model = BertClassifierTextCNNSingleModel(bert_dir=pretrained,from_pretrained=False,hidden_size=hidden_size)
        elif 'coattention' in infer_model_name:
            print('模型名称：sentence_bert_coattention')
            if 'msdrop' in infer_model_name:
                print('使用multi_sampel_drop')
                model = SBERTCoAttentionModel_MDP(bert_dir=pretrained,from_pretrained=False,hidden_size=hidden_size)
            else:
                print('不使用multi_sample_drop')
                model = SBERTCoAttentionModel(bert_dir=pretrained,from_pretrained=False,hidden_size=hidden_size)
        elif 'lstm' in infer_model_name:
            print('模型名称：Bert_lstm')
            print('注意！：使用RNN类模型时不能使用model.eval(),请检查代码中是否含有此项')
            if 'msdrop' in infer_model_name:
                print('使用multi_sampel_drop')
                model = BertLstm_MDP(bert_dir=pretrained,from_pretrained=False,hidden_size=hidden_size)
            else:
                print('不使用multi_sample_drop')
                model = BertLstm(bert_dir=pretrained,from_pretrained=False,hidden_size=hidden_size)
    elif 'nezha' in infer_model_name:
        print('使用nezha模型')
        print('模型名称：sentence_nezha')
        if 'msdrop' in infer_model_name:
            print('使用multi_sampel_drop')
            model = SNEZHASingleModel_MDP(bert_dir=pretrained,from_pretrained=False,hidden_size=hidden_size)
        else:
            print('不使用multi_sample_drop')
            model = SNEZHASingleModel(bert_dir=pretrained,from_pretrained=False,hidden_size=hidden_size)

    tokenizer = AutoTokenizer.from_pretrained(pretrained)
    test_dataset = TestData(test_data, max_length, tokenizer)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    y_pred_list = []
    #在测试集上推理模型结果
    for i in range(num_of_models):
        model_dict = torch.load(os.path.join(saved_model_path, '{}_{}.pth'.format(infer_model_name, i)))
        model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
        model.load_state_dict(model_dict)
        model.to(device)
        # os.path.join(model_path,'bert_best_{}.pth'.format(i))为保存模型的路径
        # torch.load()为把bert_0.pth读取为torch格式，然后model.load_state_dict（）加载刚刚torch格式的权重
        print('======================开始在测试集上测试！======================')
        model.eval()
        label_predict_list = np.array([], dtype=int)
        for step, batch in enumerate(tqdm(test_dataloader)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, segment_ids = batch
            with torch.no_grad():
                logit, probability = model(input_ids, segment_ids)
            label_pred = np.argmax(probability.detach().to("cpu").numpy(), axis=1)
            label_predict_list = np.concatenate([label_predict_list, label_pred])
        y_pred_list.append(label_predict_list.tolist())

    # 利用投票法融合模型
    y_pred_array = np.array(y_pred_list)
    y_pred_array = y_pred_array.T
    y_pred_final = []
    for i in trange(y_pred_array.shape[0]):
        count = np.bincount(y_pred_array[i])
        label = np.argmax(count)
        y_pred_final.append(label)

    # 写入json文件
    with open(output_dir + 'Pred_{}.json'.format(output_filename), 'w+') as f:
        #labels_list = [{'label':str(label)} for label in list(label_predict_list)]
        for label in list(y_pred_final):
            f.write(json.dumps({'label': int(label)}) + '\n')

    # 写入csv文件
    data_ = pd.DataFrame(y_pred_final)
    data_.to_csv(output_dir + 'Pred_{}.csv'.format(model_name), index=None, header=None)




if __name__ == '__main__':
    config=Config()
    num_of_inferred_models = config.num_of_inferred_models
    hidden_size = config.hidden_size
    saved_model_path = config.saved_model_path
    infer_model_name = config.infer_model_name
    output_dir = config.output_dir
    max_length = config.max_length
    batch_size = config.batch_size
    output_filename = '{}'.format(infer_model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    test_data_path = 'test.tsv'
    test_data = pd.read_csv(test_data_path, header=None, sep='\t')
    test_data.columns = ['q1', 'q2']
    print(test_data.head())

    #文本清理,清理标点符号
    r = "[_.!+-=——,$%^，。？?、~@#￥%……&*《》<>「」{}【】()/\\\[\]]"
    for i in trange(test_data.shape[0]):
        test_data['q1'].iloc[i] = re.sub(r, '', test_data['q1'].iloc[i])
        test_data['q2'].iloc[i] = re.sub(r, '', test_data['q2'].iloc[i])

    # 选择预训练模型,只使用它们的config
    print('======================加载预训练模型======================')
    if 'roberta' in infer_model_name:
        print('使用Roberta模型')
        if 'bwcc' in infer_model_name:
            pretrained = 'uer/roberta-base-word-chinese-cluecorpussmall'
        elif 'bfcc' in infer_model_name:
            pretrained = 'uer/roberta-base-finetuned-chinanews-chinese'
        else:
            pretrained = 'hfl/chinese-roberta-wwm-ext'
    elif 'macbert' in infer_model_name:
        print('使用macbert模型')
        pretrained = 'hfl/chinese-macbert-base'
    elif 'nezha' in infer_model_name:
        print('使用nezha模型')
        pretrained = 'pretrained_model/nezha-base-www/'
    elif 'ernie' in infer_model_name:
        print('使用ernie模型')
        pretrained = 'pretrained_model/ernie-1.0/'

    infer(device,num_of_inferred_models,pretrained,hidden_size,saved_model_path,infer_model_name,output_filename,output_dir)



