import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from src.AI.preprocess import Dataset, df_test, df_train, df_val
from src.AI.model import BertClassifier

np.random.seed(112)
model = BertClassifier()
EPOCHS = 1
LR = 1e-6


def train(
    model, train_data, val_data, learning_rate, epochs, batch_size=2, checkpoint=None
):
    # Initialize the training and validation datasets
    train, val = Dataset(train_data), Dataset(val_data)

    # Initialize the data loaders for the training and validation datasets
    # The dataloaders will provide batches of data to the model during training.
    train_dataloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True
    )
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=batch_size)

    # Check if a GPU is available and if not, use a CPU
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Initialize the loss function and the optimizer
    # CrossEntropyLoss is often used in multi-class classification problems
    # Adam is a popular choice of optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)

    # If a GPU is available, move the model and loss function to the GPU
    if use_cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    # Initialize the global step counter
    global_step = 0

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("model sucsessfully loaded: \n")

    for epoch_num in range(epochs):
        # Initialize accumulators for the total training accuracy and loss
        total_acc_train = 0
        total_loss_train = 0

        # Iterate over the batches of the training data loader
        for train_input, train_label in tqdm(train_dataloader):
            # Move the labels and inputs to the GPU if available
            train_label = train_label.to(device)
            mask = train_input["attention_mask"].to(device)
            input_id = train_input["input_ids"].squeeze(1).to(device)

            # Pass the inputs through the model
            output = model(input_id, mask)

            # Calculate the loss of the model's predictions against the true labels
            batch_loss = criterion(output, train_label.long())
            total_loss_train += batch_loss.item()

            # Calculate the accuracy of the model's predictions
            acc = (output.argmax(dim=1) == train_label).sum().item()
            total_acc_train += acc

            # Reset the gradients of the model parameters
            model.zero_grad()
            # Perform backpropagation to calculate the gradients
            batch_loss.backward()
            # Update the model parameters
            optimizer.step()

            global_step += 1
            if global_step % 500 == 0:
                torch.save(
                    {
                        "epoch": epoch_num,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": total_loss_train,
                    },
                    f"src/AI/checkpoints/checkpoint_4_{global_step}.pt",
                )

        # Initialize accumulators for the total validation accuracy and loss
        total_acc_val = 0
        total_loss_val = 0

        with torch.no_grad():
            for val_input, val_label in val_dataloader:
                # Move the labels and inputs to the GPU if available
                val_label = val_label.to(device)
                mask = val_input["attention_mask"].to(device)
                input_id = val_input["input_ids"].squeeze(1).to(device)

                # Pass the inputs through the model
                output = model(input_id, mask)

                # Calculate the loss of the model's predictions against the true labels
                batch_loss = criterion(output, val_label.long())
                total_loss_val += batch_loss.item()

                # Calculate the accuracy of the model's predictions
                acc = (output.argmax(dim=1) == val_label).sum().item()
                total_acc_val += acc

            print(
                f"Epochs: {epoch_num + 1} | Train Loss: {total_loss_train / len(train_data): .3f} | Train Accuracy: {total_acc_train / len(train_data): .3f} | Val Loss: {total_loss_val / len(val_data): .3f} | Val Accuracy: {total_acc_val / len(val_data): .3f}"
            )
            torch.save(
                {
                    "epoch": epoch_num,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": total_loss_train,
                },
                f"src/AI/checkpoints/checkpoint_4_{global_step}F.pt",
            )


def main():
    response = str(input("Resume Training? (Y/N): \n"))
    if response == "Y" or response == "y":
        print(len(df_train), len(df_val), len(df_test))
        train(
            model,
            df_train,
            df_val,
            LR,
            EPOCHS,
            checkpoint="src/AI/checkpoints/checkpoint_4_890F.pt",
        )
    elif response == "N" or response == "n":
        print(len(df_train), len(df_val), len(df_test))
        train(
            model,
            df_train,
            df_val,
            LR,
            EPOCHS,
        )
    else:
        print("Invalid Response")


if __name__ == "__main__":
    main()
