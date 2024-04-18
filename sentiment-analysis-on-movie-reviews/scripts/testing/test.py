import sys
import time
import pickle
import argparse
import torch
import os
import pandas as pd

from torch.utils.data import DataLoader

sys.path.append('../../model')
from data import MyDataset
from model import Mymodel
from mutils import correct_calulations


def test(model, dataloader, mode = "normal"):
    """
    Test the accuracy of a model on some labelled test dataset.

    Args:
        model: The torch module on which testing must be performed.
        dataloader: A DataLoader object to iterate over some dataset.

    Returns:
        batch_time: The average time to predict the classes of a batch.
        total_time: The total time to process the whole dataset.
        accuracy: The accuracy of the model on the input data.
    """
    # Switch the model to eval mode.
    model.eval()
    device = model.device

    time_start = time.time()
    batch_time = 0.0
    final_labels = []

    # Deactivate autograd for evaluation.
    with torch.no_grad():
        for batch in dataloader:
            batch_start = time.time()

            # Move input and output data to the GPU if one is used.
            sentence = batch["sentence"].to(device)
            sentence_length = batch["sentence_length"].to(device)
            labels = batch["label"].to(device)

            _, probs = model(sentence,
                             sentence_length,
                             model = mode)

            final_labels += correct_calulations(probs)
            batch_time += time.time() - batch_start

    batch_time /= len(dataloader)
    total_time = time.time() - time_start

    return batch_time, total_time, final_labels


def main(test_file, pretrained_file, batch_size=32, mode="normal"):
    """
    Test the model with pretrained weights on some dataset.

    Args:
        test_file: The path to a file containing preprocessed data.
        pretrained_file: The path to a checkpoint produced by the
            'train_model' script.
        vocab_size: The number of words in the vocabulary of the model
            being tested.
        embedding_dim: The size of the embeddings in the model.
        hidden_size: The size of the hidden layers in the model. Must match
            the size used during training. Defaults to 300.
        num_classes: The number of classes in the output of the model. Must
            match the value used during training. Defaults to 3.
        batch_size: The size of the batches used for testing. Defaults to 32.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for testing ", 20 * "=")

    checkpoint = torch.load(pretrained_file, map_location=device)

    # Retrieving model parameters from checkpoint.
    vocab_size = checkpoint["model"]["_word_embedding.weight"].size(0)
    embedding_dim = checkpoint["model"]['_word_embedding.weight'].size(1)
    hidden_size = 300
    num_classes = 5

    print("\t* Loading test data...")
    with open(test_file, "rb") as pkl:
        test_data_past = pickle.load(pkl)
        test_data = MyDataset(test_data_past, max_length = 50, state = "test")

    test_loader = DataLoader(test_data, shuffle=False, batch_size=batch_size)

    print("\t* Building model...")
    model = Mymodel(vocab_size,
                 embedding_dim,
                 hidden_size,
                 num_classes=num_classes,
                 device=device).to(device)

    model.load_state_dict(checkpoint["model"])

    print(20 * "=",
          " Testing model on device: {} ".format(device),
          20 * "=")
    batch_time, total_time, labels = test(model, test_loader, mode = mode)

    print("-> Average batch processing time: {:.4f}s, total test time:\
 {:.4f}s, labels: {}".format(batch_time, total_time, len(labels)))
    
    heads = ['PhraseId', 'Sentiment']
    labels = [i.item() for i in labels]
    test_data_PhraseId = test_data_past['PhraseIds']
    results = pd.DataFrame({'PhraseId': test_data_PhraseId, 'Sentiment': labels})
    results.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model on\
 some dataset")
    parser.add_argument("--test_data",
                        default="../../data/preprocessed/test_data.pkl",
                        help="Path to a file containing preprocessed test data")
    parser.add_argument("--checkpoint",
                        default="../../data/checkpoints/best.pth.tar",
                        help="Path to a checkpoint with a pretrained model")
    parser.add_argument("--model",
                        default="normal",
                        help="Model waited to be selected")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size to use during testing")
    args = parser.parse_args()

    main(args.test_data,
         args.checkpoint,
         args.batch_size,
         mode = args.model)
