import sys
import pandas as pd
import torch
import numpy as np

from src.AI.preprocess import Dataset
from src.AI.model import BertClassifier


class_names = ["business", "entertainment", "sport", "tech", "politics"]


def run():
    context = input(
        "Enter 1 to classify article, enter 2 to test model with a dataframe: \n"
    )

    print(context == "1")
    if context == "1" or context == "2":
        m = BertClassifier()
        context = int(context)
        check = torch.load("src/AI/models/model4.pt")
        m.load_state_dict(check)
        m.eval()
        test_input(m) if context == 1 else test_file(m)
    else:
        print("Invalid input")
        sys.exit(0)


def test_file(m):
    df = pd.read_csv("data/test-1.csv")
    test = Dataset(df)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        m = m.cuda()
    for test_input, test_label in test_dataloader:
        test_label = test_label.to(device)
        mask = test_input["attention_mask"].to(device)
        input_id = test_input["input_ids"].squeeze(1).to(device)
        output = m(input_id, mask)
        _, predicted_indices = torch.max(output, 1)

        predicted_classes = [class_names[index] for index in predicted_indices]

        print(f"The article is catagorized as: {predicted_classes[0]} \n")


def test_input(m):
    context = str(input("Enter article text: \n"))
    data = {"category": ["sport"], "text": [context]}
    df = pd.DataFrame.from_dict(data)
    test = Dataset(df)
    test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        m = m.cuda()
    for test_input, test_label in test_dataloader:
        test_label = test_label.to(device)
        mask = test_input["attention_mask"].to(device)
        input_id = test_input["input_ids"].squeeze(1).to(device)
        output = m(input_id, mask)
        _, predicted_indices = torch.max(output, 1)

        predicted_classes = [class_names[index] for index in predicted_indices]

        print(f"The article is catagorized as: {predicted_classes[0]} \n")


if __name__ == "__main__":
    run()
