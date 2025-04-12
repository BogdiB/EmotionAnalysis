# emotion_analysis.py

import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Set stopwords
stop_words = set(stopwords.words('english'))

# -------------------------------
# Step 1: Load Dataset
# -------------------------------
def load_data(file_path):
    df = pd.read_csv(file_path)
    if 'text' not in df.columns or 'emotion' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'emotion' columns")
    return df


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
    le = LabelEncoder()
    df['emotion_label'] = le.fit_transform(df['emotion'])
    return df, le


# -------------------------------
# Step 4: Train/Test Split
# -------------------------------
def split_data(df):
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'], df['emotion_label'], test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


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
    clf = LogisticRegression()
    clf.fit(X_train_vec, y_train)
    return clf


# -------------------------------
# Step 7: Evaluate Model
# -------------------------------
def evaluate_model(clf, X_test_vec, y_test, label_encoder):
    y_pred = clf.predict(X_test_vec)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# -------------------------------
# Main Function
# -------------------------------
def main():
    print("Loading dataset...")
    df = load_data("emotion_dataset.csv")
    
    print("Cleaning and preprocessing...")
    df, label_encoder = preprocess(df)
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)
    
    print("Vectorizing text...")
    X_train_vec, X_test_vec, vectorizer = vectorize_text(X_train, X_test)
    
    print("Training model...")
    clf = train_model(X_train_vec, y_train)
    
    print("Evaluating model...")
    evaluate_model(clf, X_test_vec, y_test, label_encoder)


if __name__ == "__main__":
    main()
