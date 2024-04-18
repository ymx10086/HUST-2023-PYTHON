import torch
import torch.nn as nn
import warnings
 
warnings.filterwarnings('ignore')

from layers import RNNDropout, PositionalEncoding, SelfAttention, MaskLM, MLMHead, Bert, SiBert, TextCNN
from config import Config 
import torch.nn.functional as F

class Mymodel(nn.Module):
    """
    Implementation of the model.
    """

    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_size,
                 embeddings=None,
                 padding_idx=0,
                 dropout=0.5,
                 num_classes=5,
                 device="cpu",
                 mode = "train"):
        """
        Args:
            vocab_size: The size of the vocabulary of embeddings in the model.
            embedding_dim: The dimension of the word embeddings.
            hidden_size: The size of all the hidden layers in the network.
            embeddings: A tensor of size (vocab_size, embedding_dim) containing
                pretrained word embeddings. If None, word embeddings are
                initialised randomly. Defaults to None.
            padding_idx: The index of the padding token in the sentences
                passed as input to the model. Defaults to 0.
            dropout: The dropout rate to use between the layers of the network.
                A dropout rate of 0 corresponds to using no dropout at all.
                Defaults to 0.5.
            num_classes: The number of classes in the output of the network.
                Defaults to 3.
            device: The name of the device on which the model is being
                executed. Defaults to 'cpu'.
        """
        super(Mymodel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        self.device = device
        self.config = Config()
        self.mode = mode

        self._masklm = MaskLM(self.config, mlm_probability=0.25)

        self._mlmhead = MLMHead(self.config, self.hidden_size, self.vocab_size)

        # self._bert = Bert(self.config, mode=mode)

        self._word_embedding = nn.Embedding(self.vocab_size,
                                            self.embedding_dim,
                                            padding_idx=padding_idx,
                                            _weight=embeddings)
        
        self._textcnn = TextCNN(self.config)
                                        
        self._dense = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size)
        )
            
        self._positional_encoding = PositionalEncoding(embedding_dim,
                                                       dropout=0.2,
                                                       max_len=100)

        if self.dropout:
            self._rnn_dropout = RNNDropout(p=self.dropout)
            # self._rnn_dropout = nn.Dropout(p=self.dropout)
        
        self._lstm = nn.LSTM(self.embedding_dim,
                             self.hidden_size,
                             num_layers = 1,
                             bias = True,
                             batch_first=True,
                             dropout=dropout,
                             bidirectional=True)
        
        self._GRU = nn.GRU(self.embedding_dim,
                            self.hidden_size,
                            num_layers = 1,
                            bias = True,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=True)

        self._selfattention = SelfAttention(self.config, self.hidden_size)
        
        self._turn = nn.Linear(self.hidden_size * 2, self.hidden_size)

        self._classification = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size * 50,
                                                       self.num_classes))
        
        self._classification1 = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size * 2,
                                                       self.num_classes))
        
        self._classification2 = nn.Sequential(nn.Dropout(p=self.dropout),
                                             nn.Tanh(),
                                             nn.Dropout(p=self.dropout),
                                             nn.Linear(self.hidden_size,
                                                       self.num_classes))

        # self._sibert = SiBert(self.config.checkpoint)
        if mode == 'pretrain':
            self.lm = MaskLM(self.config)
            self.mlm_head = MLMHead(self.config, self.hidden_size, self.vocab_size)

        # Initialize all weights and biases in the model.
        self.apply(_init_esim_weights)

    def forward(self,
                sentences,
                sentences_lengths,
                model = None):
        """
        Args:
            sentences: A batch of varaible length sequences of word indices
                representing premises. The batch is assumed to be of size
                (batch, sentences_lengths).
            sentences_lengths: A 1D tensor containing the lengths of the
                sentences in 'sentences'.
            model: A string incading the state of the training model.Default to
                None.

        Returns:
            logits: A tensor of size (batch, num_classes) containing the
                logits for each output class of the model.
            probabilities: A tensor of size (batch, num_classes) containing
                the probabilities of each output class in the model.
        """
        if model == "normal":
            # ipdb.set_trace()
            if self.mode == 'pretrain':
                device = sentences.device
                input_ids_masked, lm_label = self.lm.torch_mask_tokens(sentences.cpu())
                sentences = input_ids_masked.to(device)
                lm_label = lm_label[:, 1:].to(device)
            sentences = self._word_embedding(sentences)

            sentences = self._rnn_dropout(sentences)
            sentences = self._dense(sentences)
            
            sentences, (h_n, c_n) = self._GRU(sentences)
            sentences = self._turn(sentences)
            sentences = self._positional_encoding(sentences)
            attention_mask = torch.ones(sentences.shape[0], sentences.shape[1]).to(self.device)
            sentences = self._selfattention(sentences, attention_mask)
            
            if self.mode == "pretrain":
                lm_prediction_scores = self.mlm_head(sentences)[:, 1:, :]
                # ipdb.set_trace()
                pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
                lm_loss = F.cross_entropy(pred, lm_label.contiguous().view(-1))
                return lm_loss

            sentences = sentences.contiguous().view(sentences.size()[0], -1)
            
            logits = self._classification(sentences)
            probabilities = nn.functional.softmax(logits, dim=-1)
            return logits, probabilities
        
        elif model == "testcnn":
            sentences = self._word_embedding(sentences)
            sentences = self._rnn_dropout(sentences)
            # sentences = self._dense(sentences)
            sentences = self._textcnn(sentences)
            sentences = sentences.contiguous().view(sentences.size()[0], -1)
    
            logits = self._classification2(sentences)
            probabilities = nn.functional.softmax(logits, dim=-1)
            return logits, probabilities
        
        elif model == "bilstm":
            if self.mode == 'pretrain':
                device = sentences.device
                input_ids_masked, lm_label = self.lm.torch_mask_tokens(sentences.cpu())
                sentences = input_ids_masked.to(device)
                lm_label = lm_label[:, 1:].to(device)
            sentences = self._word_embedding(sentences)
            sentences = self._rnn_dropout(sentences)
            # sentences = self._dense(sentences)
            
            sentences, (h_n, c_n) = self._lstm(sentences)
            
            if self.mode == "pretrain":
                lm_prediction_scores = self.mlm_head(sentences)[:, 1:, :]
                pred = lm_prediction_scores.contiguous().view(-1, self.vocab_size)
                lm_loss = F.cross_entropy(pred, lm_label.contiguous().view(-1))
                return lm_loss
            
            sentences = sentences.contiguous().view(sentences.size()[0], -1)
            
            logits = self._classification1(sentences)
            probabilities = nn.functional.softmax(logits, dim=-1)
            return logits, probabilities
        
        elif model == "bert":
            if self.mode == "pretrain":
                attention_mask = torch.ones(sentences.shape[0], sentences.shape[1])
                loss = self._bert(sentences, attention_mask)
                return loss
            attention_mask = torch.ones(sentences.shape[0], sentences.shape[1])
            logits, probabilities = self._bert(sentences, attention_mask)
            return logits, probabilities
        else:
            attention_mask = torch.ones(sentences.shape[0], sentences.shape[1]).to(device)
            out = self._sibert(sentences, attention_mask)
            logits = out.logits
            probabilities = nn.functional.softmax(logits, dim=-1)
            return logits, probabilities

# init in proper and excellent way, which can be learned from the
def _init_esim_weights(module):
    """
    Initialise the weights of the Mymodel.
    """
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)

    elif isinstance(module, nn.LSTM):
        nn.init.xavier_uniform_(module.weight_ih_l0.data)
        nn.init.orthogonal_(module.weight_hh_l0.data)
        nn.init.constant_(module.bias_ih_l0.data, 0.0)
        nn.init.constant_(module.bias_hh_l0.data, 0.0)
        hidden_size = module.bias_hh_l0.data.shape[0] // 4
        module.bias_hh_l0.data[hidden_size:(2*hidden_size)] = 1.0

        if (module.bidirectional):
            nn.init.xavier_uniform_(module.weight_ih_l0_reverse.data)
            nn.init.orthogonal_(module.weight_hh_l0_reverse.data)
            nn.init.constant_(module.bias_ih_l0_reverse.data, 0.0)
            nn.init.constant_(module.bias_hh_l0_reverse.data, 0.0)
            module.bias_hh_l0_reverse.data[hidden_size:(2*hidden_size)] = 1.0

