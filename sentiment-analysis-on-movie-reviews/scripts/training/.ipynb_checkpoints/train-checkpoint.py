import os
import argparse
import pickle
import torch
import json
import ipdb
import sys
import warnings
 
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import torch.nn as nn
from torchviz import make_dot

from torch.utils.data import DataLoader

sys.path.append('../../model')
from data import MyDataset
from model import Mymodel

from utils import train, validate


def main(train_file,
         embeddings_file,
         target_dir,
         hidden_size=300,
         dropout=0.5,
         num_classes=3,
         epochs=30,
         batch_size=32,
         lr=0.0004,
         patience=5,
         max_grad_norm=10.0,
         checkpoint=None,
         mode = None,
         pretrain = None,
         use_pgd = False):
    """
    Train the model on the Sentiment dataset.

    Args:
        train_file: A path to some preprocessed data that must be used
            to train the model.
        valid_file: A path to some preprocessed data that must be used
            to validate the model.
        embeddings_file: A path to some preprocessed word embeddings that
            must be used to initialise the model.
        target_dir: The path to a directory where the trained model must
            be saved.
        hidden_size: The size of the hidden layers in the model. Defaults
            to 300.
        dropout: The dropout rate to use in the model. Defaults to 0.5.
        num_classes: The number of classes in the output of the model.
            Defaults to 3.
        epochs: The maximum number of epochs for training. Defaults to 64.
        batch_size: The size of the batches for training. Defaults to 32.
        lr: The learning rate for the optimizer. Defaults to 0.0004.
        patience: The patience to use for early stopping. Defaults to 5.
        checkpoint: A checkpoint from which to continue training. If None,
            training starts from scratch. Defaults to None.
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(20 * "=", " Preparing for training ", 20 * "=")

    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # -------------------- Data loading ------------------- #
    print("\t* Loading training data...")
    if mode != "sibert": 
        with open(train_file, "rb") as pkl:
            x = pickle.load(pkl)
            train_data = MyDataset(x, max_length = 50, state = "Train")
            valid_data = MyDataset(x, max_length = 50, state = "Valid")

        train_loader = DataLoader(train_data, shuffle=True, batch_size=32, num_workers = 10)
        valid_loader = DataLoader(valid_data, shuffle=True, batch_size=32, num_workers = 10)
    else:
        with open(train_file, "rb") as pkl:
            x = pickle.load(pkl)
            train_data = MyDataset(x, max_length = 50, state = "Train")
            valid_data = MyDataset(x, max_length = 50, state = "Valid")

    # -------------------- Model definition ------------------- #
    print("\t* Building model...")
    with open(embeddings_file, "rb") as pkl:
        embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)\
                     .to(device)

    model = Mymodel(embeddings.shape[0],
                 embeddings.shape[1],
                 hidden_size,
                 embeddings=embeddings,
                 dropout=dropout,
                 num_classes=num_classes,
                 device=device,
                 mode="train").to(device)

    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())


    # -------------------- Preparation for training  ------------------- #
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                           mode="max",
                                                           factor=0.5,
                                                           patience=0)

    best_score = 0.0
    start_epoch = 1

    # Data for loss curves plot.
    epochs_count = []
    train_losses = []
    valid_losses = []

    # Continuing training from a checkpoint if one was given as argument.
    # whether download pretrained model
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint["epoch"] + 1

        print("\t* Training will continue on existing model from epoch {}..."
              .format(start_epoch))

        model.load_state_dict(checkpoint["model"], False)
        if pretrain != "pretrain":
            best_score = checkpoint["best_score"]
            optimizer.load_state_dict(checkpoint["optimizer"])
            epochs_count = checkpoint["epochs_count"]
            train_losses = checkpoint["train_losses"]
            valid_losses = checkpoint["valid_losses"]


    # -------------------- Training epochs ------------------- #
    print("\n",
          20 * "=",
          "Training model on device: {}".format(device),
          20 * "=")

    patience_counter = 0
    for epoch in range(start_epoch, epochs+1):
        epochs_count.append(epoch)

        print("* Training epoch {}:".format(epoch))
        epoch_time, epoch_loss, epoch_accuracy = train(model,
                                                       train_loader,
                                                       optimizer,
                                                       criterion,
                                                       epoch,
                                                       max_grad_norm,
                                                       use_pgd = use_pgd,
                                                       mode=mode
                                                      )
        
        # Compute loss and accuracy before starting (or resuming) training.
        _, valid_loss, valid_accuracy = validate(model,
                                                 valid_loader,
                                                 criterion,
                                                 mode=mode
                                                )
        print("\t* Validation loss before training: {:.4f}, accuracy: {:.4f}%"
          .format(valid_loss, (valid_accuracy*100)))
        
        valid_losses.append(valid_loss)
        train_losses.append(epoch_loss)
        print("-> Training time: {:.4f}s, loss = {:.4f}, accuracy: {:.4f}%"
              .format(epoch_time, epoch_loss, (epoch_accuracy*100)))

        # Update the optimizer's learning rate with the scheduler.
        scheduler.step(epoch_accuracy)

        # Early stopping on validation accuracy.
        if epoch_accuracy < best_score:
            patience_counter += 1
        else:
            best_score = epoch_accuracy
            patience_counter = 0
            # Save the best model. The optimizer is not saved to avoid having
            # a checkpoint file that is too heavy to be shared. To resume
            # training from the best model, use the 'Sentiment_*.pth.tar'
            # checkpoints instead.
            torch.save({"epoch": epoch,
                        "model": model.state_dict(),
                        "best_score": best_score,
                        "epochs_count": epochs_count,
                        "train_losses": train_losses,
                        "valid_losses": valid_losses},
                       os.path.join(target_dir, "best.pth.tar"))

        # Save the model at each epoch.
        torch.save({"epoch": epoch,
                    "model": model.state_dict(),
                    "best_score": best_score,
                    "optimizer": optimizer.state_dict(),
                    "epochs_count": epochs_count,
                    "train_losses": train_losses,
                    "valid_losses": valid_losses},
                   os.path.join(target_dir, "Sentiment_{}.pth.tar".format(epoch)))

        if patience_counter >= patience:
            print("-> Early stopping: patience limit reached, stopping...")
            break

    # Plotting of the loss curves for the train and validation sets.
    plt.figure()
    plt.plot(epochs_count, train_losses, "-r")
    plt.plot(epochs_count, valid_losses, "-b")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend(["Training loss", "Validation loss"])
    plt.title("Cross entropy loss")
    
    plt.savefig('test_{}.png'.format(mode), bbox_inches='tight')

    plt.show()



if __name__ == "__main__":
    default_config = "../../config/training.json"

    parser = argparse.ArgumentParser(description="Train the model on dataset")
    parser.add_argument("--config",
                        default=default_config,
                        help="Path to a json configuration file")
    parser.add_argument("--checkpoint",
                        default=None,
                        help="Path to a checkpoint file to resume training")
    parser.add_argument("--pretrain",
                        default=None,
                        help="Judge the way to resume training")
    parser.add_argument("--model",
                        default="normal",
                        help="Model waited to be selected")
    parser.add_argument("--use_pgd",
                        default=True,
                        help="Judge to use pgd to better trainig")
    
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.realpath(__file__))

    if args.config == default_config:
        config_path = os.path.join(script_dir, args.config)
    else:
        config_path = args.config

    with open(os.path.normpath(config_path), 'r') as config_file:
        config = json.load(config_file)

    main(
        os.path.normpath(os.path.join(script_dir, config["train_data"])),
        #  os.path.normpath(os.path.join(script_dir, config["valid_data"])),
         os.path.normpath(os.path.join(script_dir, config["embeddings"])),
         os.path.normpath(os.path.join(script_dir, "../../../autodl-tmp")),
         config["hidden_size"],
         config["dropout"],
         config["num_classes"],
         # config["epochs"],
        20,
         config["batch_size"],
         0.0004,
         config["patience"],
         config["max_gradient_norm"],
         args.checkpoint,
         mode = args.model,
         pretrain = args.pretrain,
         use_pgd = args.use_pgd
    )
