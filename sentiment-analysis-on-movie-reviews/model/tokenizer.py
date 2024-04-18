import os
from nltk import tokenize
from collections import defaultdict
from config import Config


SPECIAL_TOKEN = {
    'pad': '[PAD]',
    'cls': '[CLS]',
    'sep': '[SEP]',
    'mask': '[MASK]',
    'unk': '[UNK]'
}


class Vocab:
    def __init__(self) -> None:
        self.token_list = []
        self.token_to_idx = {}
        self.__token_freqs = defaultdict(int)
        self.init_special_token()

    def init_special_token(self):
        self.all_special_ids = []
        self.all_special_tokens = []
        for name, token in SPECIAL_TOKEN.items():
            self.__setattr__(name+'_token', token)
            self.__setattr__(name+'_token_id', len(self.token_list))
            self.token_to_idx[token] = len(self.token_list)
            self.all_special_ids.append(len(self.token_list))
            self.token_list.append(token)
            self.all_special_tokens.append(token)

    def update_token_freqs(self, tokens):
        for token in tokens:
            self.__token_freqs[token] += 1

    def build(self, min_freq=1):
        for token, freq in self.__token_freqs.items():
            if freq >= min_freq:
                self.token_to_idx[token] = len(self.token_list)
                self.token_list.append(token)

    def save_vocab(self, path):
        with open(path, 'w') as f:
            for token in self.token_list:
                f.write(token+'\n')

    def load_vocab(self, path):
        self.token_list = []
        self.token_to_idx = {}
        with open(path, 'r') as f:
            for token in f.readlines():
                self.token_to_idx[token.strip()] = len(self.token_list)
                self.token_list.append(token.strip())

    def convert_token_to_idx(self, token):
        return self.token_to_idx.get(token, self.unk_token_id)

    def convert_idx_to_token(self, idx):
        return self.token_list[idx]

    def get_all_special_ids(self):
        return self.all_special_ids
    
    def get_all_special_tokens(self):
        return self.all_special_tokens

    def __len__(self):
        return len(self.token_list)


class TokenizerEN():
    def __init__(self, config: Config, build_vocab=True) -> None:
        self.config = config
        self.vocab = Vocab()
        self.tokenizer = tokenize.WordPunctTokenizer()
        if build_vocab:
            self.build_vocab()
        else:
            self.vocab.load_vocab(config.vocab_path)
        self.init_special_token()

    def init_special_token(self):
        for name in SPECIAL_TOKEN.keys():
            self.__setattr__(
                name+'_token', self.vocab.__getattribute__(name+'_token'))
            self.__setattr__(
                name+'_token_id', self.vocab.__getattribute__(name+'_token_id'))
    
    def __len__(self):
        return len(self.vocab)

    def __call__(self, text: str, max_len=None):
        tokens = self.tokenize(text.lower())
        idx = [self.vocab.cls_token_id]
        idx.extend(self.convert_tokens_to_ids(tokens))
        idx.append(self.vocab.sep_token_id)
        att = [1]*len(idx)
        if max_len is not None:
            length=len(idx)
            if length > max_len:
                idx = idx[:max_len]
                idx[-1] = self.vocab.sep_token_id
                att = att[:max_len]
            else:
                idx.extend([self.vocab.pad_token_id]*(max_len-length))
                att.extend([0]*(max_len-length))
        return {
            'input_ids': idx,
            'attention_mask': att
        }

    def build_vocab(self):
        def _build(path):
            fname_list = os.listdir(path)
            for fname in fname_list:
                with open(os.path.join(path, fname), 'r') as f:
                    text = f.readline()
                    tokens = self.tokenize(text.lower())
                    self.vocab.update_token_freqs(tokens)
        data_path = self.config.data_path
        _build(os.path.join(data_path, 'train/pos'))
        _build(os.path.join(data_path, 'train/neg'))
        _build(os.path.join(data_path, 'train/unsup'))
        _build(os.path.join(data_path, 'test/pos'))
        _build(os.path.join(data_path, 'test/neg'))
        self.vocab.build(min_freq=self.config.min_freq)
        self.vocab.save_vocab(self.config.vocab_path)

    def tokenize(self, text):
        return self.tokenizer.tokenize(text)

    def convert_tokens_to_ids(self, tokens):
        return [self.vocab.convert_token_to_idx(token) for token in tokens]

    def convert_idx_to_toknes(self, idxs):
        return [self.vocab.convert_idx_to_token(idx) for idx in idxs]

    def get_all_special_ids(self):
        return self.vocab.all_special_ids
    
    def get_all_special_tokens(self):
        return self.vocab.all_special_tokens

    def get_special_tokens_mask(self, tokens):
        all_special_ids = self.get_all_special_ids()
        special_tokens_mask = [
            1 if token in all_special_ids else 0 for token in tokens]
        return special_tokens_mask


