import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
subject_lines = [
    "Congratulations, you won a free lottery ticket!",
    "Meeting agenda for tomorrow",
    "Get cheap meds now",
    "Your invoice is attached",
    "Win big money!!!",
    "Let's catch up next week",
    "Exclusive deal just for you",
    "Project deadline reminder",
    "Claim your free reward today",
    "Lunch at 1?"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = spam, 0 = not spam

# Train model
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(subject_lines)
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

# App title
st.title("📧 Spam Subject Line Filter")

# Input from user
user_input = st.text_input("Enter an email subject line:")

if user_input:
    # Predict
    user_vector = vectorizer.transform([user_input])
    prediction = model.predict(user_vector)[0]
    prediction_proba = model.predict_proba(user_vector)[0]

    # Show result
    if prediction == 1:
        st.error(f"🔴 Prediction: Spam (Confidence: {prediction_proba[1]*100:.2f}%)")
    else:
        st.success(f"🟢 Prediction: Not Spam (Confidence: {prediction_proba[0]*100:.2f}%)")

# Show model accuracy
if st.checkbox("Show model accuracy on test data"):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    st.write(f"✅ Model Accuracy: **{accuracy*100:.2f}%** on test data")
