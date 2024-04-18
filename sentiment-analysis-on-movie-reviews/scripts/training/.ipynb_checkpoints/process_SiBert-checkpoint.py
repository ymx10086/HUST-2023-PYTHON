import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
import argparse
import fnmatch
import json
import ipdb
import sys

sys.path.append('../../model')
from layers import PGD


from datasets import load_dataset
from datasets import load_metric

from transformers import AutoTokenizer, DataCollatorWithPadding
from transformers import AutoModelForSequenceClassification
from transformers import AdamW
from transformers import get_scheduler

import torch
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

def process_SiBert(inputdir,
                         embeddings_file,
                         targetdir,
                         lowercase=False,
                         ignore_punctuation=False,
                         num_words=None,
                         stopwords=[],
                         labeldict={},
                         bos=None,
                         eos=None,
                         use_pgd=False):
    """
    Preprocess the data from the Sentiment dataset so it can be used by the
    SiBert model.
    Compute a worddict from the train set, and transform the words in
    the sentences of the corpus to their indices, as well as the labels.
    Build an embedding matrix from pretrained word vectors.
    The preprocessed data is saved in pickled form in some target directory.

    Args:
        inputdir: The path to the directory containing the corpus.
        embeddings_file: The path to the file containing the pretrained
            word vectors that must be used to build the embedding matrix.
        targetdir: The path to the directory where the preprocessed data
            must be saved.
        lowercase: Boolean value indicating whether to lowercase the sentences
            in the input data. Defautls to False.
        ignore_punctuation: Boolean value indicating whether to remove
            punctuation from the input data. Defaults to False.
        num_words: Integer value indicating the size of the vocabulary to use
            for the word embeddings. If set to None, all words are kept.
            Defaults to None.
        stopwords: A list of words that must be ignored when preprocessing
            the data. Defaults to an empty list.
        bos: A string indicating the symbol to use for beginning of sentence
            tokens. If set to None, bos tokens aren't used. Defaults to None.
        eos: A string indicating the symbol to use for end of sentence tokens.
            If set to None, eos tokens aren't used. Defaults to None.
    """
    if not os.path.exists(targetdir):
        os.makedirs(targetdir)

    # Retrieve the train, dev and test data files from the dataset directory.
    train_file = "train.tsv/train.tsv"
    dev_file = ""
    test_file = "test.tsv/test.tsv"

    train=pd.read_csv(os.path.join(inputdir, train_file),sep='\t')
    test=pd.read_csv(os.path.join(inputdir, test_file),sep='\t')
    
    tokenizer = AutoTokenizer.from_pretrained('/root/sentiment-roberta-large-english/sibert')
    model = AutoModelForSequenceClassification.from_pretrained('/root/sentiment-roberta-large-english/sibert', num_labels=5, ignore_mismatched_sizes=True)

    def tokenize_function(example):
        return tokenizer(example, truncation=True)
    
    # -------------------- Train data preprocessing -------------------- #

    tokenized_datasets = train["Phrase"].map(tokenize_function)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    sent_list = train["Sentiment"].to_list()

    for i in range(len(tokenized_datasets)):
        tokenized_datasets[i]['label'] = sent_list[i]

    print("\t* Saving result...")
    with open(os.path.join(targetdir, "train_data_sibert.pkl"), "wb") as pkl_file:
        pickle.dump(tokenized_datasets, pkl_file)
    with open(os.path.join(targetdir, "train_data_collator.pkl"), "wb") as pkl_file:
        pickle.dump(data_collator, pkl_file)

    train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=64, collate_fn=data_collator)
    
    # -------------------- Train data  -------------------- #
    optimizer = AdamW(model.parameters(), lr=3e-5)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    num_epochs = 2
    num_training_steps = len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    
    if use_pgd:
        pgd = PGD(model)
        K = 3
    print("# -------------------- Train model ~~~ -------------------- #")
    for epoch in range(num_epochs):
        batch_index = 0
        progress_bar = tqdm(range(num_training_steps))
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            
            if use_pgd:
                pgd.backup_grad()
                for t in range(K):
                    pgd.attack(is_first_attack=(t == 0))  
                    if t != K - 1:
                        model.zero_grad()
                    else:
                        pgd.restore_grad()
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward() 
                pgd.restore()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            
            description = "Avg. batch proc.loss: {:.4f}".format(loss/(batch_index+1))
            progress_bar.set_description(description)
            batch_index = batch_index + 1
     
    
    # -------------------- Test data preprocessing -------------------- #
    tokenized_datasets_test = test["Phrase"].map(tokenize_function)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    print("\t* Saving result...")
    with open(os.path.join(targetdir, "test_data_sibert.pkl"), "wb") as pkl_file:
        pickle.dump(tokenized_datasets_test, pkl_file)
    with open(os.path.join(targetdir, "test_data_collator.pkl"), "wb") as pkl_file:
        pickle.dump(data_collator, pkl_file)
        
    test_dataloader = DataLoader(tokenized_datasets_test, batch_size=64, collate_fn=data_collator)
        
    model.eval()
    test_predictions = list()

    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)

        test_predictions.extend(predictions)
    test_predictions = [i.item()  for i in test_predictions]
    
    submission = pd.DataFrame(list(zip(test['PhraseId'], test_predictions)), columns =['PhraseId', 'Sentiment'])
    submission.to_csv('submission.csv', index=False)
    print("The submission profile has been saved.")


if __name__ == "__main__":
    default_config = "../../config/preprocessing.json"

    parser = argparse.ArgumentParser(description="Preprocess the Sentiment dataset")
    parser.add_argument(
        "--config",
        default=default_config,
        help="Path to a configuration file for preprocessing datasets"
    )
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))
    print(script_dir)

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), "r") as cfg_file:
        config = json.load(cfg_file)

    process_SiBert(
        os.path.normpath(os.path.join(script_dir, "../../data/dataset")),
        os.path.normpath(os.path.join(script_dir, config["embeddings_file"])),
        os.path.normpath(os.path.join(script_dir, config["target_dir"])),
        lowercase=config["lowercase"],
        ignore_punctuation=config["ignore_punctuation"],
        num_words=config["num_words"],
        stopwords=config["stopwords"],
        labeldict=config["labeldict"],
        bos=config["bos"],
        eos=config["eos"],
        use_pgd=True
    )
