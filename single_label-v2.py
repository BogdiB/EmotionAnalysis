# goemotions_rf_svm.py

from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC

# -------------------------------
# Choose Classifier: Set to True for SVM, False for Random Forest
USE_SVM = False
# -------------------------------

# Load dataset
print("Loading GoEmotions (simplified)...")
dataset = load_dataset("go_emotions", "simplified")
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Extract single labels
train_df['label'] = train_df['labels'].apply(lambda x: x[0])
test_df['label'] = test_df['labels'].apply(lambda x: x[0])

# Encode labels
le = LabelEncoder()
train_df['label_enc'] = le.fit_transform(train_df['label'])
test_df['label_enc'] = le.transform(test_df['label'])

# Vectorize text
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2)  # Adds bigrams for improvement
)
X_train_vec = vectorizer.fit_transform(train_df['text'])
X_test_vec = vectorizer.transform(test_df['text'])
y_train = train_df['label_enc']
y_test = test_df['label_enc']

# Train model
def train_model(X_train_vec, y_train):
    if USE_SVM:
        print("Training Support Vector Machine (SVM)...")
        clf = LinearSVC(class_weight='balanced', max_iter=1000)
    else:
        print("Training Random Forest...")
        clf = RandomForestClassifier(n_estimators=100, class_weight='balanced', n_jobs=-1)
    
    clf.fit(X_train_vec, y_train)
    return clf

clf = train_model(X_train_vec, y_train)

# Evaluate
print("Evaluating model...")
y_pred = clf.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le.classes_.astype(str)
))
