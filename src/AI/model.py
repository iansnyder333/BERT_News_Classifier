import pandas as pd
import torch
import numpy as np
from transformers import BertModel
from torch import nn


class BertClassifier(nn.Module):
    # Initialize BERT Classifier
    def __init__(self, dropout=0.5):
        # Extend the superclass for pre trained BERT Classifier
        super(BertClassifier, self).__init__()

        # Initialize the BERT model. "bert-base-cased" is a pre-trained BERT model, and we are using it to get the benefits of Transfer Learning.
        self.bert = BertModel.from_pretrained("bert-base-cased")

        # Initialize dropout layer: a dropout layer randomly drops out (by setting to zero) a number of output features of the layer during training.
        self.dropout = nn.Dropout(dropout)

        # Initialize a Linear layer: this layer combines input data into a single output through a linear transformation.
        # The linear layer's input dimension matches the output dimension of the BERT model (768), and the output dimension is 5.
        self.linear = nn.Linear(768, 5)

        # Initialize a ReLU (Rectified Linear Unit) activation function: this function will be applied to the output of the linear layer.
        self.relu = nn.ReLU()

    # Define forward pass
    def forward(self, input_id, mask):
        # Pass the input to the BERT model. The BERT model returns the last layer's hidden-state of the first token of the sequence (CLS token) and a "pooled" output (an aggregation of the last layer's hidden state)
        _, pooled_output = self.bert(
            input_ids=input_id, attention_mask=mask, return_dict=False
        )
        # Pass the "pooled" output through the dropout layer
        dropout_output = self.dropout(pooled_output)

        # Pass the output of the dropout layer to the linear layer
        linear_output = self.linear(dropout_output)

        # Apply the ReLU activation function to the output of the linear layer
        final_layer = self.relu(linear_output)

        # Return the output of the final layer
        return final_layer
