import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from torch import nn
from torch.optim import Adam
from tqdm import tqdm

from preprocess import Dataset
from model import BertClassifier

df = pd.read_csv("data/bbc-text.csv")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
labels = {"business": 0, "entertainment": 1, "sport": 2, "tech": 3, "politics": 4}


def evaluate(model, test_data):
    test = Dataset(test_data)

    test_dataloader = torch.utils.data.DataLoader(test, batch_size=2)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if use_cuda:
        model = model.cuda()

    total_acc_test = 0
    with torch.no_grad():
        for test_input, test_label in test_dataloader:
            test_label = test_label.to(device)
            mask = test_input["attention_mask"].to(device)
            input_id = test_input["input_ids"].squeeze(1).to(device)

            output = model(input_id, mask)

            acc = (output.argmax(dim=1) == test_label).sum().item()
            total_acc_test += acc

    print(f"Test Accuracy: {total_acc_test / len(test_data): .3f}")


if __name__ == "__main__":
    np.random.seed(112)
    df_train, df_val, df_test = np.split(
        df.sample(frac=1, random_state=42), [int(0.8 * len(df)), int(0.9 * len(df))]
    )
    m = BertClassifier()
    """
    checkpoing = torch.load(
        "/Users/iansnyder/Desktop/Projects/NER_proj/src/AI/models/model4.pt"
    )
    m.load_state_dict(checkpoing)
    m.eval()
    evaluate(m, df_test)
    """
