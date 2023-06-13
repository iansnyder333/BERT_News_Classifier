import pandas as pd
import torch
import numpy as np


from preprocess import Dataset
from model import BertClassifier


def test_file(m, path):
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
    m = BertClassifier()

    class_names = ["business", "entertainment", "sport", "tech", "politics"]
    check = torch.load("src/AI/models/model4.pt")
    m.load_state_dict(check)
    m.eval()

    test_file(m, "data/test-1.csv")
