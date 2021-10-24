class Config():
    def __init__(self):

        self.model_name = 'roberta_bfcc_sbert_coattention_fl_lr'
        self.saved_model_path = '/home/zhangruichang/sg2021/baseline/BERT/saved_model/macbert_Sbert_coatten'
        #模型参数
        self.max_length = 64
        self.hidden_size = 768
        #训练参数
        self.seed = 1227
        self.epochs = 4
        self.batch_size = 64
        self.lr = 2e-5
        self.weight_decay = 1e-3
        self.num_warmup_steps = 2000

        #infer
        self.output_dir = 'results'
        self.num_of_inferred_models = 5
        self.infer_model_name = 'bert_best_macbert_sbert_coattention_msdrop_fl_lr'