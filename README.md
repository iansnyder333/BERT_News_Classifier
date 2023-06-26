# BERT_News_Classifier
NLP to classify news articles into catagories.
Model inputs a news article as raw text and outputs one of the following categories: "business", "entertainment", "sport", "tech", or "politics".

## Demo Video 



https://github.com/iansnyder333/BERT_News_Classifier/assets/58576523/a7af66e6-888f-4c2d-8694-402ce6c9461d




## Table of Contents

- [How to Download](#how-to-download)
- [How to Run](#how-to-run)
- [Notes](#notes)

## How to Download

```sh
git clone https://github.com/iansnyder333/BERT_News_Classifier.git
cd BERT_News_Classifier
python3.11 -m venv venv
source venv/bin/activate
pip3.11 install -r requirements.txt
```

## How to Run

```sh
python3.11 main.py
```

## Notes

Python 3.11 required.

Model was trained with 5 EPOCHS using bbc-text csv file from kaggle. Model has 97% accuracy on testing data but is still in development for accuratley classifying new articles.
