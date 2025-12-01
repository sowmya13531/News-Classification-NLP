# 📰 Fake News Classification Using Gensim Word2Vec Vectors

This project builds a Fake vs. Real News classifier using pretrained Gensim Word2Vec word embeddings and a Gradient Boosting Classifier from scikit-learn.
The workflow includes data preprocessing using spaCy, vectorization using Google News Word2Vec embeddings, model training, evaluation, and testing on custom inputs.

## 🚀 Project Overview

This project performs:
Loading Dataset
Fake/Real news dataset (fake_and_real_news.csv)
Text Preprocessing
Tokenization, lemmatization, stopword & punctuation removal via spaCy (en_core_web_lg)
Conversion to numerical embeddings using Gensim Word2Vec Google News vectors (300-dim)

## Model Training

Train-test split (80/20) with stratification

Model: GradientBoostingClassifier

### Evaluation

Accuracy, precision, recall, F1-score
Confusion matrix visualization
Prediction on New Articles

## 📦 Installation and Setup

Install dependencies:

```bash
pip install gensim
```
```bash
python -m spacy download en_core_web_lg
```

### Load pretrained Google News vectors (~1.6 GB):

```bash
import gensim.downloader as api
wv = api.load('word2vec-google-news-300')
```

### 📑 Dataset

Load the dataset:
```bash
import pandas as pd
df = pd.read_csv('/content/fake_and_real_news.csv')
```


### Dataset structure:

Text → news article
label → Fake or Real

### Convert labels:

df['label_num'] = df.label.map({'Fake': 0, 'Real': 1})

🧹 Text Preprocessing & Vectorization

### Load spaCy model:

```bash
import spacy
nlp = spacy.load('en_core_web_lg')
```

## 📁 Clone the Repository

```bash
git clone https://github.com/sowmya13531/News-Classification-NLP/Gensim-Word-Vectors/.git
cd News-Classification-NLP/Gensim-Word-Vectors
```
*or*

Simply run the colab Notebook.

