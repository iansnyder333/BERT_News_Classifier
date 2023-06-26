import tkinter as tk
from tkinter import filedialog, Text, messagebox
import sys
import pandas as pd
import torch
import numpy as np

from src.AI.preprocess import Dataset
from src.AI.model import BertClassifier

# from preprocess import Dataset
# from model import BertClassifier

class_names = ["business", "entertainment", "sport", "tech", "politics"]


def classify_text():
    m = BertClassifier()
    check = torch.load("src/AI/models/model4.pt")
    m.load_state_dict(check)
    m.eval()

    context = text_box.get("1.0", "end")
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

        messagebox.showinfo(
            "Classification Result",
            f"The article is categorized as: {predicted_classes[0]}",
        )


def new_article():
    text_box.delete("1.0", tk.END)


root = tk.Tk()
root.title("Article Classifier by Ian Snyder")
label = tk.Label(
    text="Please enter the article's contents in the textbox below:",
)
label.pack()
text_box = Text(root, height=25, width=100)
text_box.pack()

classify_text_button = tk.Button(
    root, text="Classify Article", padx=10, pady=5, command=classify_text
)
classify_text_button.pack()

new_article_button = tk.Button(
    root, text="Clear Textbox", padx=10, pady=5, command=new_article
)
new_article_button.pack()
quit_button = tk.Button(root, text="Quit", padx=10, pady=5, command=root.destroy)
quit_button.pack()

root.mainloop()
