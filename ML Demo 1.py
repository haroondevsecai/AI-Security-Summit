# Step 1: Import ML Tools
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re

# Step 2: Training Data (Spam vs. Ham)
texts = [
    "Win a free iPhone now!",
    "Limited time offer!",
    "Hello, how are you?",
    "Meet me at 5pm"
]
labels = ["Spam", "Spam", "Ham", "Ham"]

# Step 3: Convert text to numbers (Vectorization)
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(texts)

# Step 4: Train the Model
model = MultinomialNB()
model.fit(X_train, labels)

# Sanitise Input Function
def sanitize_input(text):
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    return cleaned.strip()[:280]  # Limit length

# Step 5: Test the Model with Sanitisation
new_texts = ["Free Toy!", "See you tomorrow!"]
new_texts = [sanitize_input(t) for t in new_texts]

X_test = vectorizer.transform(new_texts)
predictions = model.predict(X_test)

# Step 6: Show Results
for text, prediction in zip(new_texts, predictions):
    print(f"Message: '{text}' → Predicted: {prediction}")

