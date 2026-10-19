import joblib

model = joblib.load("sentiment_model.joblib")
print("Sentiment classifier ready. Type a review (or 'quit' to exit).")

while True:
    text = input("\nReview: ")
    if text.strip().lower() == "quit":
        break
    label = model.predict([text])[0]
    confidence = model.predict_proba([text]).max()
    print(f"Prediction: {label} (confidence {confidence:.0%})")