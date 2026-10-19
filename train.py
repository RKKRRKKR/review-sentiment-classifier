import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. Load the cleaned data
df = pd.read_csv("reviews_clean.csv")

# 2. Split: 80% to learn from, 20% held back for testing
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"]
)
print("Training rows:", len(X_train), "| Test rows:", len(X_test))

# 3. Build and train the model
model = make_pipeline(
    TfidfVectorizer(ngram_range=(1, 2)),
    LogisticRegression(max_iter=1000)
)
model.fit(X_train, y_train)

# 4. Evaluate on the unseen test set
predictions = model.predict(X_test)
print("\nAccuracy:", round(accuracy_score(y_test, predictions), 3))
print("\n", classification_report(y_test, predictions))

# 5. Show the mistakes
print("Mistakes:")
for text, true, pred in zip(X_test, y_test, predictions):
    if true != pred:
        print(f"- '{text}'  (you labeled: {true}, model said: {pred})")

# 6. Save the trained model to a file
joblib.dump(model, "sentiment_model.joblib")
print("\nSaved sentiment_model.joblib")