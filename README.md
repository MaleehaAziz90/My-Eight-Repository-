# My-Eight-Repository-
 Build a Fake News Detection System using Machine Learning
# ============================================================
# File: train_model.py
# Fake News Detection System using Machine Learning
# ============================================================

# Import Libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ============================================================
# Download NLTK Resources
# ============================================================

nltk.download('stopwords')
nltk.download('wordnet')

# ============================================================
# Load Dataset
# ============================================================

df = pd.read_csv("dataset.csv")

print("\n========== DATASET PREVIEW ==========\n")
print(df.head())

# ============================================================
# Text Preprocessing
# ============================================================

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\\S+", "", text)

    # Remove punctuation
    text = text.translate(
        str.maketrans('', '', string.punctuation)
    )

    # Remove numbers
    text = re.sub(r'\\d+', '', text)

    # Tokenization
    words = text.split()

    # Remove stopwords and lemmatize
    cleaned_words = []

    for word in words:

        if word not in stop_words:

            lemma = lemmatizer.lemmatize(word)

            cleaned_words.append(lemma)

    return " ".join(cleaned_words)

# Apply preprocessing
df["clean_text"] = df["text"].apply(clean_text)

print("\n========== CLEANED DATA ==========\n")
print(df[["text", "clean_text"]].head())

# ============================================================
# Feature Engineering
# ============================================================

vectorizer = TfidfVectorizer(max_features=5000)

X = vectorizer.fit_transform(df["clean_text"])

y = df["label"]

# ============================================================
# Train Test Split
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape :", X_test.shape)

# ============================================================
# Models
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(),
    "Naive Bayes": MultinomialNB(),
    "Random Forest": RandomForestClassifier()
}

results = {}

# ============================================================
# Train & Evaluate Models
# ============================================================

for model_name, model in models.items():

    print(f"\n========== {model_name} ==========\n")

    # Train model
    model.fit(X_train, y_train)

    # Prediction
    predictions = model.predict(X_test)

    # Evaluation Metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label="Real"
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="Real"
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="Real"
    )

    # Store Accuracy
    results[model_name] = accuracy

    # Print Metrics
    print("Accuracy :", round(accuracy, 2))
    print("Precision:", round(precision, 2))
    print("Recall   :", round(recall, 2))
    print("F1-Score :", round(f1, 2))

    # Classification Report
    print("\nClassification Report:\n")

    print(classification_report(y_test, predictions))

    # Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    plt.figure(figsize=(5, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues'
    )

    plt.title(f"{model_name} - Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.show()

# ============================================================
# Select Best Model
# ============================================================

best_model_name = max(results, key=results.get)

print("\n========== BEST MODEL ==========\n")
print("Best Model:", best_model_name)

best_model = models[best_model_name]

# ============================================================
# Save Model & Vectorizer
# ============================================================

joblib.dump(best_model, "model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully!")

# ============================================================
# Feature Importance (Bonus)
# ============================================================

if best_model_name == "Logistic Regression":

    feature_names = vectorizer.get_feature_names_out()

    coefficients = best_model.coef_[0]

    top_indices = coefficients.argsort()[-10:]

    top_words = [feature_names[i] for i in top_indices]

    top_scores = [coefficients[i] for i in top_indices]

    plt.figure(figsize=(8, 5))

    plt.barh(top_words, top_scores)

    plt.title("Top Important Words")

    plt.xlabel("Importance Score")

    plt.show()