#把每个模型最终在验证集上的推理(infer)结果，用来探索最优的融合方案

import numpy as np
from sklearn.metrics import f1_score
from itertools import combinations
from tqdm import tqdm



def merge_on_valid(model_names):
    #len of valid: 9000
    total_pred_ = [0]*9000
    threshold = len(model_names)/2
#将所有模型的预测相加，和模型总数的一半作对比
#(相当于，对于某条待预测数据，超过一半的模型给出的结果是1，则该数据的预测取1)
    ##预测相加
    for model in model_names:
        print("processing model {}".format(model))
        preds_ = np.load('{}_pred_.npy'.format(model))
        preds_ = preds_.tolist()
        assert len(total_preds_) == len(preds_)
        for idx, pred_ in enumerate(preds_):
            total_preds_[idx] += pred_
    ##和总数的一半作对比，得到投票结果(vote_)
    total_preds_ = np.array(total_preds_)
    vote_ = (total_preds_>threshold).astype('int')
    gt_ = np.load(valid_dir + 'gt_.npy')

    f1 = f1_score(gt_, vote_)
    print("voted_f1:{}".format(f1))

    return f1

if __name__ == '__main__':
    valid_dir = '/home/zhangruichang/sg2021/baseline/BERT/valid_output/'
    total_model_names=[######]
    total_model_dir = [valid_dir + model_name for model_name in total_model_names]
    f1 = merge_on_valid(total_model_dir) #将模型全部融合的效果

#探索部分融合(融合3个、5个、7个、9个、11个)模型的效果
    for size in [3,5,7,9,11]:
        print("searching the best merge of {} models".format(size))
        records = []
        combs = combinations(total_model_dir, size)
        best_f1 = 0
        best_comb = None
        for comb in tqdm(combs):
            f1 = merge_on_valid(list(comb))
            if f1 > best_f1:
                best_f1 = f1
                best_comb = comb
            records.append((list(comb), f1))
        print("best f1 and model list:")
        print(best_f1, best_comb)
        merge_on_valid(list(best_comb), True)

        print("top5 candidates list:")
        records.sort(key=lambda x:x[-1], reverse=True)
        for i in range(5):
            print(records[i])






