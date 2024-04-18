import torch.nn as nn
import torch
from torch.autograd import Variable
import math
from config import Config
import torch.nn.functional as F
import pickle
import copy
from transformers.models.bert.modeling_bert import (
    BertModel, BertConfig, BertOnlyMLMHead
)

from datasets import load_dataset
from datasets import load_metric

from transformers import AutoModelForSequenceClassification

import warnings
 
warnings.filterwarnings('ignore')

class RNNDropout(nn.Dropout):
    """
    Dropout layer for the inputs of RNNs.

    Apply the same dropout mask to all the elements of the same sequence in
    a batch of sequences of size (batch, sequences_length, embedding_dim).
    """

    def forward(self, sequences_batch):
        """
        Apply dropout to the input batch of sequences.

        Args:
            sequences_batch: A batch of sequences of vectors that will serve
                as input to an RNN.
                Tensor of size (batch, sequences_length, emebdding_dim).
                #torch.Size([32, 61, 300])
        Returns:
            A new tensor on which dropout has been applied.
        """
        # torch.Size([32, 300])
        ones = sequences_batch.data.new_ones(sequences_batch.shape[0],
                                             sequences_batch.shape[-1])
        dropout_mask = nn.functional.dropout(ones, self.p, self.training,
                                             inplace=False)
        # torch.Size([32, 300])
        return dropout_mask.unsqueeze(1) * sequences_batch

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], 
                         requires_grad=False)
        return self.dropout(x)

class SelfAttention(nn.Module):
    def __init__(self, config: Config, hidden_size) -> None:
        super(SelfAttention, self).__init__()
        self.hidden_size = hidden_size

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = hidden_size//config.num_attention_heads
        self.all_head_size = self.num_attention_heads*self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.dropout1 = nn.Dropout(config.dropout_prob)
        self.linear = nn.Linear(self.all_head_size, self.hidden_size)
        self.dropout2 = nn.Dropout(config.dropout_prob)

    def transpose(self, x: torch.Tensor):
        #[bs,length,hidden_size]->[bs,num_heads,length,att_hidden_size]
        new_shape = x.size()[:-1]+(self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask : torch.Tensor):
        key_layer = self.transpose(self.key(hidden_states))
        value_layer = self.transpose(self.value(hidden_states))
        quert_layer = self.transpose(self.query(hidden_states))
        attention_scores = torch.matmul(quert_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        attention_mask = attention_mask[:, None, None, :]
        attention_mask = (1-attention_mask)*(-10000.0)
        attention_scores = attention_scores + attention_mask
        attention_probs = F.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout1(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_shape = context_layer.size()[:-2]+(self.all_head_size,)
        context_layer = context_layer.view(new_shape)
        return self.dropout2(self.linear(context_layer))

class FeedForward(nn.Module):
    def __init__(self, config: Config, hidden_size) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Dropout(config.dropout_prob),
            nn.Linear(hidden_size, config.feed_forward_size),
            nn.GELU(),
            nn.Dropout(config.dropout_prob),
            nn.Linear(config.feed_forward_size, hidden_size)
        )

    def forward(self, x):
        return self.dense(x)

class TextCNN(nn.Module):
    def __init__(self, config: Config, kersize=[3, 5, 7, 9], hidden_size = 300) -> None:
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden_size, config.cnn_hidden_size, kernel_size=k),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(1)
            ) for k in kersize])
        self.classifier = nn.Sequential(
            nn.Linear(config.cnn_hidden_size*len(kersize), hidden_size),
            nn.Dropout(config.dropout_prob)
        )

    def forward(self, inputs):
        embeds = inputs.permute(0, 2, 1)
        out = [conv(embeds) for conv in self.convs]
        out = torch.concat(out, dim=1)
        out = out.view(-1, out.shape[1])
        out = self.classifier(out)
        return out

class PGD():
    def __init__(self, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}

    def attack(self, epsilon=1., alpha=0.3, emb_name='_word_embedding.weight', is_first_attack=False):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0:
                    r_at = alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, epsilon)

    def restore(self, emb_name='_word_embedding.weight'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]
                
# def get_model_path_list(base_dir):
#     """
#     从文件夹中获取 model.pt 的路径
#     """
#     model_lists = []

#     for fname in os.listdir(base_dir):
#         if 'Sentiment' in fname: 
#             model_lists.append(base_dir+fname)

#     model_lists = sorted(model_lists)
    
#     return model_lists[:]

# def SWA(model,base_dir='../../data/checkpoints'):
#     """
#     swa 滑动平均模型，一般在训练平稳阶段再使用 SWA
#     """
#     model_path_list = get_model_path_list(base_dir)
#     print(f'mode_list:{model_path_list}')

#     swa_model = copy.deepcopy(model)
#     swa_n = 0.

#     with torch.no_grad():
#         start_epoch, best_score = 0, 0
#         for _ckpt in model_path_list:
            
#             checkpoint = torch.load(checkpoint)
#             start_epoch = max(start_epoch, checkpoint["epoch"] + 1)
#             best_score = max(best_score, checkpoint["best_score"])

#             model.load_state_dict(checkpoint["model"])
#             optimizer.load_state_dict(checkpoint["optimizer"])
#             epochs_count = checkpoint["epochs_count"]
#             train_losses = checkpoint["train_losses"]
#             valid_losses = checkpoint["valid_losses"]
#             checkpoint = torch.load(_ckpt)

#             tmp_para_dict = dict(model.named_parameters())

#             alpha = 1. / (swa_n + 1.)

#             for name, para in swa_model.named_parameters():
#                 para.copy_(tmp_para_dict[name].data.clone() * alpha + para.data.clone() * (1. - alpha))

#             swa_n += 1

#     # use 100000 to represent swa to avoid clash


#         torch.save({"epoch": epoch,
#                 "model": model.state_dict(),
#                 "best_score": best_score,
#                 "optimizer": optimizer.state_dict(),
#                 "epochs_count": epochs_count,
#                 "train_losses": train_losses,
#                 "valid_losses": valid_losses},
#                os.path.join(target_dir, "swa_Sentiment_{}.pth.tar".format(epoch)))

#     return swa_model

class Embeddings(nn.Module):
    def __init__(self, config: Config, embedding_size) -> None:
        super().__init__()
        self.word_embeddings = nn.Embedding(
            config.vocab_size, embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, embedding_size)
        self.layerNorm = nn.LayerNorm(embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout_prob)

    def forward(self, input_ids: torch.Tensor, positions_ids=None):
        bs, length = input_ids.size()
        if positions_ids is None:
            positions_ids = torch.arange(length).expand((bs, -1)).to(input_ids.device)
        input_embeds = self.word_embeddings(input_ids)
        positions_embeds = self.position_embeddings(positions_ids)
        embeddings = input_embeds+positions_embeds
        embeddings = self.dropout(self.layerNorm(embeddings))
        return embeddings

class MaskLM():
    def __init__(self, config: Config, mlm_probability=0.25):
        self.mlm_probability = mlm_probability
        # self.tokenizer = TokenizerEN(config, build_vocab=False)
        with open("../../data/preprocessed/worddict.pkl", "rb") as pkl:
            self.worddict = pickle.load(pkl)

    def torch_mask_tokens(self, inputs: torch.Tensor):

        labels = inputs.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # special_tokens_mask = [
        #     self.tokenizer.get_special_tokens_mask(val) for val in labels.tolist()
        # ]
        # special_tokens_mask = torch.tensor(
        #     special_tokens_mask, dtype=torch.bool)

        probability_matrix.masked_fill_(probability_matrix, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100

        # 80% MASK
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        inputs[indices_replaced] = 2 #self.worddict["_MASK_"]

        # 10% random
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & \
            masked_indices & ~indices_replaced
        # ipdb.set_trace()
        random_words = torch.randint(
            len(self.worddict), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]

        # 10% original
        return inputs, labels

class MLMHead(nn.Module):
    def __init__(self, config: Config, hidden_size,  vocab_size) -> None:
        super().__init__()
        self.dense = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size, eps=config.layer_norm_eps),
            nn.Linear(hidden_size, vocab_size)
        )

    def forward(self, sequence_output):
        return self.dense(sequence_output)

class Bert(nn.Module):
    def __init__(self, config: Config, mode = 'train') -> None:
        super().__init__()
        bert_config = BertConfig.from_pretrained(
            config.bert_name, cache_dir=config.bert_cache)
        bert = BertModel.from_pretrained(
            config.bert_name, cache_dir=config.bert_cache)
        self.embeddings = Embeddings(config, bert_config.hidden_size)
        
        self.encoder = copy.deepcopy(bert.encoder)

        self.mode = mode
        if mode == 'pretrain':
            bert_config.vocab_size = config.vocab_size
            self.vocab_size = config.vocab_size
            self.mlm_head = BertOnlyMLMHead(bert_config)
            self.lm = MaskLM(config)

        self.classifier = nn.Linear(bert_config.hidden_size, 5)

    def forward(self, inputs, attention_mask):
        if self.mode == 'pretrain':
            device = inputs.device
            input_ids_masked, lm_label = self.lm.torch_mask_tokens(inputs.cpu())
            inputs = input_ids_masked.to(device)
            lm_label = lm_label[:, 1:].to(device)

        inputs_embeds = self.embeddings(inputs)
        mask_expanded = attention_mask[:, None, None, :]
        mask_expanded = (1-mask_expanded)*(-10000.0)

        outputs = self.encoder(
            inputs_embeds, attention_mask=mask_expanded)['last_hidden_state']
        
        if self.mode == 'pretrain':
            lm_prediction_scores = self.mlm_head(outputs)[:, 1:, :]
            pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
            lm_loss = F.cross_entropy(pred, lm_label.contiguous().view(-1))
            return lm_loss

        sentences = outputs[:, 0, :]
        # mask = attention_mask.unsqueeze(-1).expand(outputs.shape).float()
        # sum_embeds = torch.sum(outputs*mask, dim=1)
        # sum_mask = torch.sum(mask, dim=1).clamp(min=1e-9)
        # features_mean = sum_embeds/sum_mask
        logits = self.classifier(sentences)
        probabilities = nn.functional.softmax(logits, dim=-1)

        return logits, probabilities

class SiBert(nn.Module):
    def __init__(self, checkpoint):
        super(SiBert, self).__init__()
        self._model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=5, ignore_mismatched_sizes=True)
    def forward(self, sentences, attention_mask):
        return self._model(sentences, attention_mask)

