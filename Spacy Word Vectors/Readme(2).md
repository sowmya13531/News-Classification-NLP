# 📰 Fake vs Real News Classification using SpaCy Word Vectors

This project demonstrates text classification using SpaCy’s pre-trained word vectors (en_core_web_lg).
The goal is to classify news articles as Fake or Real based on their semantic meaning captured through word embeddings

## 📌 Project Overview

Dataset contains 9,900 news headlines labelled as Fake or Real.
Used SpaCy Large English Model (en_core_web_lg) to generate 300-dimensional word vectors for each text.
Converted each text headline into a single dense vector using SpaCy’s doc.vector.
Trained a machine learning classifier on these vectors to distinguish between Fake and Real news.
This project is inspired by the Codebasics NLP tutorials.

### 📂 Dataset

Column	Description
Text	News headline
Label	Fake / Real
label_num	Converted label (Fake = 0, Real = 1)
Dataset size: 9,900 rows

### Label distribution:

Fake: 5000
Real: 4900
Balanced dataset → Good for classification.

### 🧠 Workflow

#### 1. Load the dataset

*df = pd.read_csv('Fake_Real_Data.csv')*


#### 2. Preprocess labels

*df['label_num'] = df['label'].map({'Fake': 0, 'Real': 1})*

#### 3. Download and load SpaCy model

📦 Installation and Setup
Install dependencies:

```bash
pip install spacy
python -m spacy download en_core_web_lg
```

```bash import spacy
nlp = spacy.load('en_core_web_lg')
```

#### 4. Generate word vectors

Each text is converted into a 300-dimensional vector.

*df['vector'] = df['Text'].apply(lambda x: nlp(x).vector)*

#### 5. Train-test split

*x_train, x_test, y_train, y_test = train_test_split(
    df.vector, df.label_num, test_size=0.2, random_state=2000
)*

#### 6. Train a classifier

(You can use Logistic Regression, SVM, Random Forest, etc.) Example:

*from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=2000)
model.fit(list(x_train), y_train)*

#### 7. Evaluate the model

*y_pred = model.predict(list(x_test))
accuracy_score(y_test, y_pred)*

### 🚀 Why SpaCy Word Vectors?

Pre-trained on large corpora
Captures semantic meaning
Reduces the need for heavy deep learning
Fast and efficient for classical ML models
This approach is perfect when you want good accuracy without GPU requirements.

### 🧪 Results

(You can update this section with your actual results.)

Example:
Accuracy: 92%
Precision/Recall: High due to balanced dataset

#### 📦 Requirements

*pandas
scikit-learn
spacy
en-core-web-lg*

#### Install dependencies:

```bash 
pip install -r requirements.txt
```
### Clone the Repository
```bash
git clone https://github.com/sowmya13531/News-Classification-NLP/Spacy-Word-Vectors/.git
cd News-Classification-NLP/Spacy-Word-Vectors
```
or

Simply run the Google Colab Notebook
