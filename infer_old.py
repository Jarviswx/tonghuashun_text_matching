import os
import re
import json
import random
import torch
import logging
import numpy as np
import pandas as pd
import torch.nn as nn
from random import random
from tqdm import tqdm,trange
from config import Config
from sklearn import metrics
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig,BertTokenizer
from model import BertClassifierSingleModel, NezhaClassifierSingleModel, SBERTSingleModel, SNEZHASingleModel_MDP, BertClassifierTextCNNSingleModel_MDP, SBERTCoAttentionModel_MDP

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
class DevData(Dataset):
    def __init__(self, train_data, max_length, tokenizer):
        self.data = train_data
        self.max_length = max_length
        self.max_num_tokens = max_length - 3

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        input_ids, segment_ids, label = [], [], []
        q1 = self.data['q1'].iloc[idx]  # 获得语句q1
        q2 = self.data['q1'].iloc[idx]  # 获得语句q2
        label = self.data['label'].iloc[idx]  # 获得标签
        tokens_a = tokenizer.tokenize(q1)  # 获得在词库字典中q1的键(分词)
        tokens_b = tokenizer.tokenize(q2)  # 获得在词库字典中q2的键(分词)
        truncate_seq_pair(tokens_a, tokens_b, self.max_num_tokens)  # 限制语句长度，方便并行计算
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]  # 组合得到输入bert的规范语句
        input_ids = tokenizer.convert_tokens_to_ids(tokens)  # 将输入1向量化

        segment_ids = [0 for _ in range(len(tokens_a) + 2)] + [1 for _ in range(len(tokens_b) + 1)]
        assert len(segment_ids) == len(input_ids)

        padding_length = self.max_length - len(input_ids)
        input_ids += [0] * padding_length  # 对不足max_length的输入1补齐位数
        segment_ids += [0] * padding_length  # 对不足max_length的输入2补齐位数
        assert len(segment_ids) == len(input_ids) == self.max_length

        input_ids = torch.tensor(input_ids, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)
        return input_ids, segment_ids, label

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

def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
        if random.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()

def infer(model, device, dev_dataloader, test_dataloader, search_thres=True, threshold_fixed_=0.5, save_valid=True):
    print("Inferring")
    model.eval()
    model = torch.nn.DataParallel(model,device_ids=[0])
    model.to(device)
    total_gt_, total_preds_, total_probs_ =  [], [], []

    print("Model running on dev set...")
# search for the optimal threshold from the eval data
    for idx, batch in enumerate(tqdm(dev_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_types, label = batch

        with torch.no_grad():
            logit, probability = model(input_ids, input_types)
            output = probability#eg:[[0.1,0.5,0.4],
                                #    [0.4,0.5,0.1],
                                #    [0.1,0.4,0.5]]
            probs_ = [prob[-1] for prob in output.cpu().numpy().tolist()]
            all_gt = label.detach().to("cpu").numpy().tolist()
            all_pred = output.argmax(axis=1).cpu().numpy().tolist()
            gt_ = all_gt
            preds_ = all_pred

            total_gt_ += gt_
            total_preds_ += preds_
            total_probs_ += probs_

    # search for the optimal threshold for Binary Classification
    if search_thres:
        print("Searching for the best threshold on valid dataset...")
        thresholds = np.arange(0.2, 0.9, 0.01)
        fscore_ = np.zeros(shape=(len(thresholds)))

        print("Original F1 Score: {}".format(
            str(metrics.f1_score(total_gt_, total_preds_, zero_division=0))))
        if len(total_gt_) != 0:
            print("\tClassification Report\n")
            print(metrics.classification_report(total_gt_, total_preds_))

        for index, thres in enumerate(tqdm(thresholds)):
            y_pred_prob_ = (np.array(total_probs_) > thres).astype('int') # more than thres = 1, less than = 0
            fscore_[index] = metrics.f1_score(total_gt_, y_pred_prob_.tolist(), zero_division=0)

        # record the optimal threshold
        index_ = np.argmax(fscore_) # find the index of the max f1_score
        threshold_opt_ = round(thresholds[index_], ndigits=4) # get the optimal threshold
        f1_score_opt_ = round(fscore_[index_], ndigits=6)
        print('Best Threshold for Task: {} with F-Score: {}'.format(threshold_opt_, f1_score_opt_))

        if save_valid:
            y_pred_prob_ = (np.array(total_probs_) > threshold_opt_).astype('int')
            np.save('/home/zhangruichang/sg2021/baseline/BERT/train_as_eval_output/{}_pred_.npy'.format(model_type), y_pred_prob_)
            np.save('/home/zhangruichang/sg2021/baseline/BERT/train_as_eval_output/{}_gt.npy', np.array(total_gt_))

# inferring based on the searched optimal threshold from the eval data
    total_ids_, total_probs_ = [], []
    for idx, batch in enumerate(tqdm(test_dataloader)):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_types = batch

        with torch.no_grad():
            logit, probability = model(input_ids, input_types)
            output = probability
            all_pred = output.argmax(axis=1).cpu().numpy().tolist()
            preds_ = all_pred
            probs_ = [prob[-1] for prob in output.cpu().numpy().tolist()]

            total_preds_ += preds_
            total_probs_ += probs_

    # positive if the prob passes the original threshold of 0.5
    total_fixed_preds_ = (np.array(total_probs_) > threshold_fixed_).astype('int').tolist()

    if search_thres:
        # positive if the prob passes the optimal threshold
        total_preds_ = (np.array(total_probs_) > threshold_opt_).astype('int').tolist()
    else:
        total_preds_ = None

    return total_ids_, total_preds_, total_fixed_preds_

if __name__=='__main__':
    max_length = 64
    config = Config()
    device = config.device
    dummy_pretrained = config.dummy_pretrained

    model_type = config.infer_model_name
    model_name = config.infer_model_name
    hidden_size = config.hidden_size
    output_dir= config.infer_output_dir
    output_filename = config.infer_output_filename
    save_dir = config.save_dir
    data_dir = config.data_dir

    infer_bs = config.infer_bs
    search_thres = config.infer_search_thres
    threshold_fixed_ = config.infer_fixed_thres_

#Load test data
    test_data_path = 'test_.tsv'
    eval_data_path = '/home/zhangruichang/sg2021/baseline/BERT/Test/train_as_eval.csv'

    test_data = pd.read_csv(test_data_path, header=None, sep='\t')
    eval_data = pd.read_csv(eval_data_path)

    test_data.columns = ['q1', 'q2']
    eval_data.columns = ['label', 'q1', 'q2']
    # 文本清理,清理标点符号
    r = "[_.!+-=——,$%^，。？?、~@#￥%……&*《》<>「」{}【】()/\\\[\]]"
    for i in trange(test_data.shape[0]):
        test_data['q1'].iloc[i] = re.sub(r, '', test_data['q1'].iloc[i])
        test_data['q2'].iloc[i] = re.sub(r, '', test_data['q2'].iloc[i])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading Bert Model from {}...".format(save_dir + model_name))
    if 'sbert' in model_type.lower():
        print("Using SentenceBERT model and dataset")
        if 'nezha' in model_type.lower():
            model = SNEZHASingleModel(bert_dir=dummy_pretrained, from_pretrained=False, hidden_size=hidden_size)
        else:
            model = SBERTCoAttentionModel(bert_dir=dummy_pretrained, from_pretrained=False, hidden_size=hidden_size)

        model = torch.nn.DataParallel(model, device_ids=[0])
        model.to(device)
        model.load_state_dict(torch.load(save_dir + model_name))
        #model_dict = torch.load(save_dir + model_name)
        #model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
        #model.load_state_dict({k.replace('all_classifier', 'classifier'): v for k, v in model_dict.items()})

        #model = nn.DataParallel(model, device_ids=[0])
        #model.to(device)
        #model.load_state_dict(model_dict)

        print("Loading Dev Data...")
        tokenizer = BertTokenizer.from_pretrained(dummy_pretrained)
        dev_dataset = DevData(eval_data, max_length, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=infer_bs, shuffle=False)

        print("Loading Test Data...")
        # for test dataset, is_train should be set to False, thus get ids instead of labels
        test_dataset = TestData(test_data, max_length, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=infer_bs, shuffle=False)

    else:
        print("Using BERT model and dataset")
        if 'nezha' in model_type.lower():
            print("Using NEZHA pretrained model")
            model = NezhaClassifierSingleModel(bert_dir=dummy_pretrained, from_pretrained=False,
                                               hidden_size=hidden_size)
        elif 'cnn' in model_type.lower():
            print("Adding TextCNN after BERT output")
            model = BertClassifierTextCNNSingleModel_MDP(bert_dir=dummy_pretrained, from_pretrained=False,
                                                     hidden_size=hidden_size)
        else:
            print("Using conventional BERT model with linears")
            model = BertClassifierSingleModel(bert_dir=dummy_pretrained, from_pretrained=False, hidden_size=hidden_size)

        model_dict = torch.load(save_dir + model_name)
        model_dict = {k.replace('module.', ''): v for k, v in model_dict.items()}
        model.load_state_dict({k.replace('all_classifier', 'classifier'): v for k, v in model_dict.items()})
        #model = nn.DataParallel(model, device_ids=[0])
        #model.to(device)
        #model.load_state_dict(torch.load(save_dir + model_name))

        print("Loading Dev Data...")
        tokenizer = BertTokenizer.from_pretrained(dummy_pretrained)
        dev_dataset = DevData(eval_data, max_length, tokenizer)
        dev_dataloader = DataLoader(dev_dataset, batch_size=infer_bs, shuffle=False)

        print("Loading Test Data...")
        # for test dataset, is_train should be set to False, thus get ids instead of labels

        test_dataset = TestData(test_data, max_length, tokenizer)
        test_dataloader = DataLoader(test_dataset, batch_size=infer_bs, shuffle=False)

    total_ids_, total_preds_, total_fixed_preds_= infer(model, device, dev_dataloader, test_dataloader, search_thres, threshold_fixed_)

    #输出csv数据
    data = pd.read_csv('test_.tsv', header=None, sep='\t')
    S1 = data.iloc[:,0].to_list()
    S2 = data.iloc[:,1].to_list()
    out_csv = pd.DataFrame(np.transpose([total_preds_,S1,S2]))
    out_csv.to_csv(output_dir + 'Pred_{}.csv'.format(output_filename),index=None)

    #输出json数据
    with open(output_dir + 'Pred_{}.json'.format(output_filename), 'w+') as f:
        #     labels_list = [{'label':str(label)} for label in list(label_predict_list)]
        for label in list(total_preds_):
            f.write(json.dumps({'label': int(label)}) + '\n')


    #if total_preds_ is not None:
    #    with open(output_dir + 'opt_thre_pred_' + output_filename, 'w') as f_out:
    #        for label in list(total_preds_):
    #            f_out.write(json.dumps({'label': int(label)}) + '\n')











