from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

# Configuration
USE_SVM = False
USE_GRID_SEARCH = True

print("Loading GoEmotions (simplified)...")
dataset = load_dataset("go_emotions", "simplified")
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

# Filter to only single-label examples
train_df = train_df[train_df['labels'].apply(lambda x: len(x) == 1)].copy()
test_df = test_df[test_df['labels'].apply(lambda x: len(x) == 1)].copy()

# Use the single label
train_df['label'] = train_df['labels'].apply(lambda x: x[0])
test_df['label'] = test_df['labels'].apply(lambda x: x[0])

# Encode labels
le = LabelEncoder()
train_df['label_enc'] = le.fit_transform(train_df['label'])
test_df['label_enc'] = le.transform(test_df['label'])

# Enhanced TF-IDF
print("Vectorizing text...")
vectorizer = TfidfVectorizer(
    max_features=20000,
    min_df=3,
    sublinear_tf=True,
    stop_words='english',
    ngram_range=(1, 2)
)
X_train_vec = vectorizer.fit_transform(train_df['text'])
X_test_vec = vectorizer.transform(test_df['text'])
y_train = train_df['label_enc']
y_test = test_df['label_enc']

# Train model
def train_model(X_train_vec, y_train):
    if USE_SVM:
        print("Using Support Vector Machine (SVM)...")
        base_clf = LinearSVC(class_weight='balanced', max_iter=10000)
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100]
        }
    else:
        print("Using Random Forest...")
        base_clf = RandomForestClassifier(class_weight='balanced', n_jobs=-1, random_state=42)
        
        param_grid = {
            'n_estimators': [300],
            'max_depth': [100],
            'min_samples_split': [2],
            'min_samples_leaf': [1]
        }

    if USE_GRID_SEARCH:
        print("Performing grid search...")
        grid = GridSearchCV(base_clf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train_vec, y_train)
        print("Best parameters:", grid.best_params_)
        return grid.best_estimator_
    else:
        base_clf.fit(X_train_vec, y_train)
        return base_clf

# Train
clf = train_model(X_train_vec, y_train)

# Evaluate
print("Evaluating model...")
y_pred = clf.predict(X_test_vec)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_.astype(str)))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_, square=True)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()