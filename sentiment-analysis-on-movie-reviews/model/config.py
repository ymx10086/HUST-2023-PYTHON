import json
import torch


class Config():
    def __init__(self) -> None:

        self.seed = 42
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.data_path = './data/aclImdb/'
        self.vocab_path = './vocab/vocab.txt'
        self.log_path = './log/'
        self.save_path = './save/'
        self.bert_cache = './cache/'
        self.bert_name = 'roberta-base'

        self.model_name = 'lstm'

        self.checkpoint = '../../../sentiment-roberta-large-english/sibert'

        self.min_freq = 10
        self.max_len = 512  # max 3005 min 8 mean 282 >250 39964 >500 12408
        self.vocab_size = 39700
        self.pad_token_id = 0

        self.weight_decay = 0.01
        self.learning_rate = 5e-4 
        self.adam_epsilon = 1e-6
        self.warmup_ratio = 0.05 
        self.val_ratio = 0.01
        self.train_batch_size = 64
        self.val_batch_size = 256
        self.test_batch_size = 256
        self.pretrain_batch_size = 32
        self.max_epochs = 10
        self.print_steps = 20

        self.embedding_size = 768
        self.max_position_embeddings = 512
        self.cnn_hidden_size = 64
        self.lstm_hidden_size = 1024
        self.lstm_num_layers = 3
        self.num_attention_heads = 16
        self.feed_forward_size = 1024
        self.dropout_prob = 0.3
        self.layer_norm_eps = 1e-5
        
        self.use_ema = False
        self.use_fgm = False
        self.use_pgd = True

    def save_dict(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.__dict__, f, indent=4)

    def load_from_dict(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            dic = json.load(f)
        for key, value in dic.items():
            self.__setattr__(key,value)