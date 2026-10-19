# review-sentiment-classifier
Building a Small Sentiment Classifier

Goal. I wanted hands-on practice with the full data workflow behind a machine learning model: collecting, labeling, cleaning, training, evaluating and improving. 
I built a small text classifier in Python (pandas and scikit-learn) that labels short product reviews as positive, negative or neutral.

Data and labeling. I labeled 341 short reviews myself in three rounds. 
The review texts were AI-generated for practice, and the labels and guidelines are mine. 
I wrote a labeling rule before each round, for example: label the reviewer’s overall opinion; for mixed reviews, choose the side they end up on; sarcasm counts by its real meaning; neutral means no clear opinion or genuinely lukewarm. 
I cleaned the data (removed empty rows, duplicates and typos in labels), held out 20% for testing, and trained a TF-IDF and logistic regression model.

What I found.
My first model scored 97.1% on a small, easy test set, but that was misleading. Many examples used similar sentence patterns.
After I added harder examples (short reviews, “but” sentences, sarcasm, lukewarm wording), accuracy dropped to 88.2% on a more honest test set.
My first neutral label had only 2 examples, so the model never predicted it. After adding neutral examples, prediction for ambiguous sentiment improved.
Most errors came from words the model hadn’t seen enough (“dull”, “bad”), very short texts like “ok”, and mixed reviews where one negative word outvoted a positive ending.

Lessons.
Label quality and coverage mattered more than the model. Fixing the data helped more than changing the algorithm would have.
Borderline cases need a written rule, and the rule has to be applied consistently, or the labels become noise.
A small test set gives noisy scores. Looking at the individual mistakes taught me more than the accuracy number did.
