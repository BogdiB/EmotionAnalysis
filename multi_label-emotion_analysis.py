# emotion_analysis_multi_label.py

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report, accuracy_score, hamming_loss
from sklearn.multiclass import OneVsRestClassifier

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

# Define emotion columns manually based on dataset
EMOTION_COLUMNS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
def load_data(file_paths):
    dfs = []
    for path in file_paths:
        df = pd.read_csv(path)
        df = df[df[EMOTION_COLUMNS].sum(axis=1) > 0]  # Filter rows with at least one emotion
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

# -------------------------------
# Step 2: Clean Text
# -------------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = " ".join([word for word in text.split() if word not in stop_words])
    return text

# -------------------------------
# Step 3: Preprocess
# -------------------------------
def preprocess(df):
    df['clean_text'] = df['text'].apply(clean_text)
    labels = df[EMOTION_COLUMNS].values
    mlb = MultiLabelBinarizer(classes=EMOTION_COLUMNS)
    y = mlb.fit_transform([EMOTION_COLUMNS[i] for i in row.nonzero()[0]] for row in labels)
    return df, y, mlb

# -------------------------------
# Step 4: Split Data
# -------------------------------
def split_data(df, y):
    return train_test_split(df['clean_text'], y, test_size=0.2, random_state=42)

# -------------------------------
# Step 5: Vectorize Text
# -------------------------------
def vectorize_text(X_train, X_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    return X_train_vec, X_test_vec, vectorizer

# -------------------------------
# Step 6: Train Model
# -------------------------------
def train_model(X_train_vec, y_train):
    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000))
    clf.fit(X_train_vec, y_train)
    return clf

# -------------------------------
# Step 7: Evaluate Model
# -------------------------------
def evaluate_model(clf, X_test_vec, y_test, mlb):
    y_pred = clf.predict(X_test_vec)
    print("Hamming Loss:", hamming_loss(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=mlb.classes_, zero_division=0))

# -------------------------------
# Main
# -------------------------------
def main():
    print("Loading datasets...")
    file_paths = [
        "data/full_dataset/goemotions_1.csv",
        "data/full_dataset/goemotions_2.csv",
        "data/full_dataset/goemotions_3.csv"
    ]
    df = load_data(file_paths)

    print("Cleaning and preprocessing...")
    df, y, mlb = preprocess(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df, y)

    print("Vectorizing text...")
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)

    print("Training model...")
    clf = train_model(X_train_vec, y_train)

    print("Evaluating model...")
    evaluate_model(clf, X_test_vec, y_test, mlb)


if __name__ == "__main__":
    main()
