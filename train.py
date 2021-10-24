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
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

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

def train(model_name,pretrained,saved_model_path,device,train_data\
          ,epochs,learning_rate,num_warmup_steps,weight_decay,batch_size,max_length):
    #文本处理
    class TrainData(Dataset):
        def __init__(self, train_data, max_length, tokenizer):
            self.data = train_data
            self.max_length = max_length
            self.max_num_tokens = max_length - 3

        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, idx):
            #         input_ids,segment_ids,label = [],[],[]
            q1 = self.data['q1'].iloc[idx]
            q2 = self.data['q2'].iloc[idx]
            label = self.data['label'].iloc[idx]
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
            label = torch.tensor(label, dtype=torch.long)
            return input_ids, segment_ids, label
    #5折训练准备
    folds = KFold(n_splits=5, shuffle=True, random_state=99)
    for fold, (train_index, eval_index) in enumerate(folds.split(train_data)):
        data_train = train_data.iloc[train_index]
        data_eval = train_data.iloc[eval_index]
        tokenizer = BertTokenizer.from_pretrained(pretrained)
        train_dataset = TrainData(data_train, max_length, tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        eval_dataset = TrainData(data_eval, max_length, tokenizer)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

        #选择自定义模型（Bert类和Nezha）
        print('======================自定义模型======================')
        if 'bert' in model_name or 'sbert' in model_name:
            print('使用Bert模型')
            if 'cnn' in model_name:
                print('模型名称：bert_textcnn')
                if 'msdrop' in model_name:
                    print('使用multi_sampel_drop')
                    model = BertClassifierTextCNNSingleModel_MDP(bert_dir=pretrained, hidden_size=hidden_size)
                else:
                    print('不使用multi_sample_drop')
                    model = BertClassifierTextCNNSingleModel(bert_dir=pretrained, hidden_size=hidden_size)
            elif 'coattention' in model_name:
                print('模型名称：sentence_bert_coattention')
                if 'msdrop' in model_name:
                    print('使用multi_sampel_drop')
                    model = SBERTCoAttentionModel_MDP(bert_dir=pretrained, hidden_size=hidden_size)
                else:
                    print('不使用multi_sample_drop')
                    model = SBERTCoAttentionModel(bert_dir=pretrained, hidden_size=hidden_size)
            elif 'lstm' in model_name:
                print('模型名称：Bert_lstm')
                print('注意！：使用RNN类模型时不能使用model.eval(),请检查代码中是否含有此项')
                if 'msdrop' in model_name:
                    print('使用multi_sampel_drop')
                    model = BertLstm_MDP(bert_dir=pretrained, hidden_size=hidden_size)
                else:
                    print('不使用multi_sample_drop')
                    model = BertLstm(bert_dir=pretrained, hidden_size=hidden_size)
        elif 'nezha' in model_name:
            print('使用nezha模型')
            print('模型名称：sentence_nezha')
            if 'msdrop' in model_name:
                print('使用multi_sampel_drop')
                model = SNEZHASingleModel_MDP(bert_dir=pretrained, hidden_size=hidden_size)
            else:
                print('不使用multi_sample_drop')
                model = SNEZHASingleModel(bert_dir=pretrained, hidden_size=hidden_size)

        print('=========================================================')
        #指定需要冻结的模型层
        no_decay = ['bias', 'LayerNorm.weight']
        decayed_params = []
        undecayed_params = []
        for name, params in model.named_parameters():
            if any(n in name for n in no_decay):
                params.requires_grad = False
                undecayed_params.append(params)
            else:
                decayed_params.append(params)
        print('num of decayed params:{},undecayed params:{}'.format(len(decayed_params), len(undecayed_params)))
        model.to(device)

        # 定义损失函数（criterion）、优化器（optimizer）和学习率规划器（scheduler）
        if 'fgm' in model_name:
            print("对抗训练：FGM")
            fgm = FGM(model)
        else:
            print('对抗训练：None')

        if 'fl' in model_name:
            print("损失函数：Focal Loss")
            criterion = focal_loss()
        else:
            print("损失函数：CrossEntropyLoss")
            criterion = nn.CrossEntropyLoss()

        optimizer_params = [{'params': decayed_params,
                             'weight_decay': weight_decay},
                            {'params': undecayed_params,
                             'weight_decay': 0.0}
                            ]

        base_optimizer = RAdam(optimizer_params, lr=learning_rate)
        optimizer = Lookahead(base_optimizer)

        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_training_steps=total_steps,
            num_warmup_steps=num_warmup_steps,
        )

        best_score = 0.
        best_loss = 100000000.0

        print('======================这是第{}次训练，总共5次======================'.format(fold))
        #开始训练
        model.train()
        patience = 0
        for epoch in range(epochs):
            f1_each_epoch = 0.
            loss_each_epoch = 0.
            tq = tqdm(train_dataloader, desc="Iteration")
            f1_sum = 0
            for step, batch in enumerate(tq):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, label = batch
                logit, probability = model(input_ids, segment_ids)
                loss = criterion(probability, label)
                optimizer.zero_grad()
                loss.backward()

                if 'fgm' in model_name9:
                    fgm.attack()
                    logit, probability = model(input_ids, segment_ids)
                    # calculate the loss and BP
                    loss_adv = criterion(probability, label)
                    loss_adv.backward()
                    fgm.restore()

                optimizer.step()
                scheduler.step()

                loss_each_epoch += loss.item()

                label_pred = np.argmax(probability.detach().to("cpu").numpy(), axis=1)
                f1 = f1_score(label.detach().to("cpu").numpy(), label_pred, average='macro')
                tq.set_postfix(fold=fold, epoch=epoch, loss=loss.item() / label.size()[0], f1_score=f1)
                f1_sum += f1
            print('The average f1 score on train_set:{}'.format(f1_sum / 522))
            if loss_each_epoch < best_loss:
                print('=============The train loss is decreasing, we save the model!=============')
                print('the current loss:{},the best loss:{}'.format(loss_each_epoch, best_loss))
                best_loss = loss_each_epoch
                #torch.save(model.state_dict(), os.path.join(model_path, 'bert_{}_{}.pth'.format(model_name,fold)))

            print('======================开始在验证集上测试！======================')
            model.eval()
            label_predict_list = np.array([], dtype=int)
            for step, batch in enumerate(tqdm(eval_dataloader)):
                batch = tuple(t.to(device) for t in batch)
                input_ids, segment_ids, label = batch
                with torch.no_grad():
                    logit, probability = model(input_ids, segment_ids)
                label_pred = np.argmax(probability.detach().to("cpu").numpy(), axis=1)
                label_predict_list = np.concatenate([label_predict_list, label_pred])
            f1_each_epoch = f1_score(np.array(train_data['label'].iloc[eval_index]), label_predict_list,
                                     average='macro')
            if f1_each_epoch > best_score:
                print('=============The eval f1 score is increasing, we save the model!=============')
                print('the current f1:{},the best f1:{}'.format(f1_each_epoch, best_score))
                print('The average f1 score on train_set:{}'.format(f1_sum / 522))
                best_score = f1_each_epoch
                torch.save(model.state_dict(),
                           os.path.join(model_path, '{}_{}.pth'.format(model_name, fold)))
                patience = 0
            else:
                patience += 1
                print('the current f1:{},the best f1:{}'.format(f1_each_epoch, best_score))
                print('The average f1 score on train_set:{}'.format(f1_sum / 522))
            if patience >= 3:
                print('the eval f1 score is not improvement,we stop training!')
                break


if __name__ == '__main__':
    #配置参数
    config = Config()
    seed = config.seed
    epochs = config.epochs
    learning_rate = config.lr
    weight_decay = config.weight_decay
    num_warmup_steps = config.num_warmup_steps
    max_length = config.max_length
    batch_size = config.batch_size
    hidden_size = config.hidden_size
    model_name = config.model_name
    saved_model_path = config.saved_model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读入数据
    train_data_path = 'train.tsv'
    train_data = pd.read_csv(train_data_path, header=None, sep='\t')
    train_data.columns = ['label', 'q1', 'q2']
    print(train_data.head())

    # 文本清理,清理标点符号
    r = "[_.!+-=——,$%^，。？?、~@#￥%……&*《》<>「」{}【】()/\\\[\]]"
    for i in trange(train_data.shape[0]):
        train_data['q1'].iloc[i] = re.sub(r, '', train_data['q1'].iloc[i])
        train_data['q2'].iloc[i] = re.sub(r, '', train_data['q2'].iloc[i])

    # 选择预训练模型
    print('======================加载预训练模型======================')
    if 'roberta' in model_name:
        print('使用Roberta模型')
        if 'bwcc' in model_name:
            pretrained = 'uer/roberta-base-word-chinese-cluecorpussmall'
        elif 'bfcc' in model_name:
            pretrained = 'uer/roberta-base-finetuned-chinanews-chinese'
        else:
            pretrained = 'hfl/chinese-roberta-wwm-ext'
    elif 'macbert' in model_name:
        print('使用macbert模型')
        pretrained = 'hfl/chinese-macbert-base'
    elif 'nezha' in model_name:
        print('使用nezha模型')
        pretrained = 'pretrained_model/nezha-base-www/'
    elif 'ernie' in model_name:
        print('使用ernie模型')
        pretrained = 'pretrained_model/ernie-1.0/'
    print(pretrained)
    #进入训练
    train(model_name,pretrained,saved_model_path,device,train_data\
          ,epochs,learning_rate,num_warmup_steps,weight_decay,batch_size,max_length\
         )
