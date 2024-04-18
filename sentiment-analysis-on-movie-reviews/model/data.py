import string
import torch
import numpy as np
import pandas as pd

from collections import Counter
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from wordcloud import WordCloud


class Preprocessor(object):
    """
    Preprocessor class for Sentiment Analysis.

    The class can be used to read the datasets, build worddicts for them
    and transform them into lists of integer indices.
    """
    def __init__(self,
                 lowercase=False,
                 ignore_punctuation=False,
                 num_words=None,
                 stopwords=[],
                 labeldict={},
                 bos=None,
                 eos=None):
        """
        Args:
            lowercase: A boolean indicating whether the words in the datasets
                being preprocessed must be lowercased or not. Defaults to
                False.
            ignore_punctuation: A boolean indicating whether punctuation must
                be ignored or not in the datasets preprocessed by the object.
            num_words: An integer indicating the number of words to use in the
                worddict of the object. If set to None, all the words in the
                data are kept. Defaults to None.
            stopwords: A list of words that must be ignored when building the
                worddict for a dataset. Defaults to an empty list.
            bos: A string indicating the symbol to use for the 'beginning of
                sentence token in the data. If set to None, the token isn't
                used. Defaults to None.
            eos: A string indicating the symbol to use for the 'end of
                sentence' token in the data. If set to None, the token isn't
                used. Defaults to None.
        """
        self.lowercase = lowercase
        self.ignore_punctuation = ignore_punctuation
        self.num_words = num_words
        self.stopwords = stopwords
        self.labeldict = labeldict
        self.bos = bos
        self.eos = eos
    def read_data(self, filepath, train = True):
        """
        Read the sentences and labels from the movies reviews dataset's
        file and return them in a dictionary.

        Args:
            filepath: The path to a file containing sentences and 
                labels that must be read. 

        Returns:
            A dictionary containing two lists, one for the reviews
            and one for the labels in the input data.
        """
        PhraseIds, SentenceIds, sentences, labels = [], [], [], []
        n, l, p, maxlen, ll = 0, 0, 0, 0, 0 
        neg, pos, nel = [], [], []
        with open(filepath, "r", encoding="utf8") as input_data:

            # Translation tables to remove punctuation from strings.
            # Opereation of Defaults
            punct_table = str.maketrans({key: " "
                                         for key in string.punctuation})

            # Ignore the headers on the first line of the file.
            next(input_data)

            for line in input_data:
                line = line.strip().split("\t")

                PhraseId = line[0]
                SentenceId = line[1]
                if len(line) < 3:
                    sentence = ' '
                else:
                    sentence = line[2]
                if train:
                    label = line[3]
                else:
                    label = "hidden"

                if self.lowercase:
                    sentence = sentence.lower()

                if self.ignore_punctuation:
                    sentence = sentence.translate(punct_table)

                # Each premise and hypothesis is split into a list of words.
                sentences.append([w for w in sentence.rstrip().split()
                                 if w not in self.stopwords])
                
                if label == "hidden":
                    pass
                elif int(label) < 3:
                    neg.append(sentence)
                elif int(label) > 3:
                    pos.append(sentence)
                else:
                    nel.append(sentence)
                    
                if label == "0" or label == "1":
                    n = n + 1
                elif label == '2':
                    l = l + 1
                else:
                    p = p + 1
                    
                ll += len([w for w in sentence.rstrip().split() if w not in self.stopwords])
                maxlen = max(maxlen, len([w for w in sentence.rstrip().split() if w not in self.stopwords]))
                
                
                labels.append(label)
                PhraseIds.append(PhraseId)
                SentenceIds.append(SentenceId)
        if train:
            neg = pd.Series(neg).str.cat(sep = ' ')
            wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color="white").generate(neg)
            plt.figure(figsize=(12,10))
            plt.imshow(wordcloud, cmap="autumn", interpolation='bilinear')
            plt.axis("off")
            plt.savefig('neg.png', bbox_inches='tight')
            
            pos = pd.Series(pos).str.cat(sep = ' ')
            wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color="white").generate(pos)
            plt.figure(figsize=(12,10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.savefig('pos.png', bbox_inches='tight')
            
            nel = pd.Series(nel).str.cat(sep = ' ')
            wordcloud = WordCloud(width=1600, height=800, max_font_size=200, background_color="white").generate(nel)
            plt.figure(figsize=(12,10))
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.axis("off")
            plt.savefig('nel.png', bbox_inches='tight')
        
        plt.show()
        
        print("More Infoormation!!!")
        print(n)  
        print(l)  
        print(p)  
        print(maxlen)  
        print(ll / 156060) 
                
        return {"PhraseIds": PhraseIds,
                "SentenceIds": SentenceIds,
                "sentences": sentences,
                "labels": labels}
    def build_worddict(self, data):
        """
        Build a dictionary associating words to unique integer indices for
        some dataset. The worddict can then be used to transform the words
        in datasets to their indices.

        Args:
            data: A dictionary containing the premises, hypotheses and
                labels of some NLI dataset, in the format returned by the
                'read_data' method of the Preprocessor class.
        """
        words = []
        [words.extend(sentence) for sentence in data["sentences"]]

        counts = Counter(words)
        num_words = self.num_words
        if self.num_words is None:
            num_words = len(counts)

        self.worddict = {}

        # Special indices are used for padding, out-of-vocabulary words, and
        # beginning and end of sentence tokens.
        self.worddict["_PAD_"] = 0
        self.worddict["_OOV_"] = 1

        offset = 2
        if self.bos:
            self.worddict["_BOS_"] = 2
            offset += 1
        if self.eos:
            self.worddict["_EOS_"] = 3
            offset += 1

        for i, word in enumerate(counts.most_common(num_words)):
            self.worddict[word[0]] = i + offset

        if self.labeldict == {}:
            label_names = set(data["labels"])
            self.labeldict = {label_name: i
                              for i, label_name in enumerate(label_names)}

    def words_to_indices(self, sentence):
        """
        Transform the words in a sentence to their corresponding integer
        indices.

        Args:
            sentence: A list of words that must be transformed to indices.

        Returns:
            A list of indices.
        """
        
        indices = []
        # Include the beggining of sentence token at the start of the sentence
        # if one is defined.
        if self.bos:
            indices.append(self.worddict["_BOS_"])

        for word in sentence:
            if word in self.worddict:
                index = self.worddict[word]
            else:
                # Words absent from 'worddict' are treated as a special
                # out-of-vocabulary word (OOV).
                index = self.worddict["_OOV_"]
            indices.append(index)
        # Add the end of sentence token at the end of the sentence if one
        # is defined.
        if self.eos:
            indices.append(self.worddict["_EOS_"])

        return indices

    def indices_to_words(self, indices):
        """
        Transform the indices in a list to their corresponding words in
        the object's worddict.

        Args:
            indices: A list of integer indices corresponding to words in
                the Preprocessor's worddict.

        Returns:
            A list of words.
        """
        return [list(self.worddict.keys())[list(self.worddict.values())
                                           .index(i)]
                for i in indices]

    def transform_to_indices(self, data):
        """
        Transform the words in the dataset, as
        well as their associated labels, to integer indices.

        Args:
            data: A dictionary containing lists of sentences
                and labels, in the format returned by the 'read_data'
                method of the Preprocessor class.

        Returns:
            A dictionary containing the transformed sentences and
            labels.
        """
        transformed_data = {"PhraseIds": [],
                            "SentenceIds": [],
                            "sentences": [],
                            "labels": []}

        for i, sentence in enumerate(data["sentences"]):
            # Ignore sentences that have a label for which no index was
            # defined in 'labeldict'.
            label = data["labels"][i]
            if label not in self.labeldict and label != "hidden":
                continue

            transformed_data["PhraseIds"].append(data["PhraseIds"][i])
            transformed_data["SentenceIds"].append(data["SentenceIds"][i])

            if label == "hidden":
                transformed_data["labels"].append(-1)
            else:
                transformed_data["labels"].append(self.labeldict[label])

            indices = self.words_to_indices(sentence)
            transformed_data["sentences"].append(indices)

        return transformed_data

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.

        Args:
            embeddings_file: A file containing pretrained word embeddings.

        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.worddict:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.worddict)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self.worddict.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                if word == "_PAD_":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)

        return embedding_matrix

class MyDataset(Dataset):
    """
    Dataset class for datasets.

    The class can be used to read preprocessed datasets where the sentences
    and labels have been transformed to unique integer indices
    (this can be done with the 'preprocess_data' script in the 'scripts'
    folder of this repository).
    """

    def __init__(self,
                 data,
                 padding_idx=0,
                 max_length=None,
                 state = "Train"):
        """
        Args:
            data: A dictionary containing the preprocessed sentences
                and labels of some dataset.
            padding_idx: An integer indicating the index being used for the
                padding token in the preprocessed data. Defaults to 0.
            max_length: An integer indicating the maximum length
                accepted for the sequences in the sentences. If set to None,
                the length of the longest sentences in 'data' is used.
                Defaults to None.
            state: A string indicating the state of the dataset including 
                (Train, Valid, test).Default to "Train".
        """
        self.lengths = [len(seq) for seq in data["sentences"]]
        self.max_length = max_length
        if self.max_length is None:
            self.max_length = max(self.lengths)
        self.state = state
        if self.state == "Train":
            self.num_sequences = len(data["sentences"]) - len(data["sentences"]) // 10
        if self.state == "Valid":
            self.num_sequences = len(data["sentences"]) // 10
        if self.state == "test":
            self.num_sequences = len(data["sentences"])

        self.data = {"PhraseIds": [],
                     "SentenceIds" : [], 
                     "sentences": torch.ones((self.num_sequences,
                                             self.max_length),
                                            dtype=torch.long) * padding_idx,
                     "labels": []}
        if self.state == 'Train':
            for i, sentence in enumerate(data["sentences"]):
                if i >= self.num_sequences:
                    break
                self.data["PhraseIds"].append(data["PhraseIds"][i])
                self.data["SentenceIds"].append(data["SentenceIds"][i])

                end = min(len(sentence), self.max_length)
                self.data["sentences"][i][:end] = torch.tensor(sentence[:end])
                self.data["labels"].append(data["labels"][i])
            self.data["labels"] = torch.tensor(self.data["labels"])
            # print("sentences: %d labels: %d" % (len(self.data["sentences"]),len(self.data["labels"])))

        if self.state == 'Valid':
            cnt = 0
            for i, sentence in enumerate(data["sentences"]):
                if i < len(data["sentences"]) - self.num_sequences:
                    continue
                self.data["PhraseIds"].append(data["PhraseIds"][i])
                self.data["SentenceIds"].append(data["SentenceIds"][i])

                end = min(len(sentence), self.max_length)
                self.data["sentences"][cnt][:end] = torch.tensor(sentence[:end])
                self.data["labels"].append(data["labels"][i])
                cnt = cnt + 1
            self.data["labels"] = torch.tensor(self.data["labels"])
            # print("sentences: %d labels: %d" % (len(self.data["sentences"]),len(self.data["labels"])))


        if self.state == 'test':
            for i, sentence in enumerate(data["sentences"]):
                
                self.data["PhraseIds"].append(data["PhraseIds"][i])
                self.data["SentenceIds"].append(data["SentenceIds"][i])

                end = min(len(sentence), self.max_length)
                self.data["sentences"][i][:end] = torch.tensor(sentence[:end])
                self.data["labels"].append(data["labels"][i])
            self.data["labels"] = torch.tensor(self.data["labels"])
            # print("sentences: %d labels: %d" % (len(self.data["sentences"]),len(self.data["labels"])))


    def __len__(self):
        return self.num_sequences

    def __getitem__(self, index):
        return {"PhraseId": self.data["PhraseIds"][index],
                "SentenceId": self.data["SentenceIds"][index],
                "sentence": self.data["sentences"][index],
                "sentence_length": min(self.lengths[index],
                                      self.max_length),
                "label": self.data["labels"][index]}

