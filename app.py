import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("📩 Spam Message Detector")
st.write("Enter a message below to check if it's spam or not.")

# Input
user_input = st.text_area("Your Message:")

# Session state for feedback
if "feedback_given" not in st.session_state:
    st.session_state.feedback_given = False

# Prediction
if st.button("Check Message"):
    if user_input.strip() != "":
        data = vectorizer.transform([user_input])
        prediction = model.predict(data)

        # Store prediction result in session
        st.session_state.last_input = user_input
        st.session_state.last_prediction = prediction[0]
        st.session_state.feedback_given = False

    else:
        st.warning("Please enter a message first!")

# Show result (persist after click)
if "last_prediction" in st.session_state:
    if st.session_state.last_prediction == 1:
        st.error("🚫 This is SPAM!")
    else:
        st.success("✅ This is NOT spam")

    # Feedback section
    st.write("Was this prediction correct?")

    col1, col2 = st.columns(2)

    if col1.button("👍 Yes"):
        st.success("Thanks for your feedback!")
        st.session_state.feedback_given = True

    if col2.button("👎 No"):
        with open("feedback.txt", "a", encoding="utf-8") as f:
            f.write(st.session_state.last_input + "\n")

        st.warning("Feedback noted! This will help improve the model.")
        st.session_state.feedback_given = True

# Retrain button (optional advanced feature)
st.write("---")
if st.button("🔄 Retrain Model with Feedback"):
    import pandas as pd

    try:
        # Load feedback
        feedback_data = pd.read_csv("feedback.txt", header=None, names=["message"])
        feedback_data["label"] = 1  # assume spam for incorrect

        # Load original dataset
        original = pd.read_csv("spam.csv", encoding='latin-1')
        original = original[['v1', 'v2']]
        original.columns = ['label', 'message']
        original['label'] = original['label'].map({'ham': 0, 'spam': 1})

        # Combine datasets
        feedback_data = pd.concat([feedback_data]*50)
        combined = pd.concat([original, feedback_data])

        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.naive_bayes import MultinomialNB

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit_transform(combined['message'])
        y = combined['label']

        model = MultinomialNB()
        model.fit(X, y)

        # Save updated model
        pickle.dump(model, open("model.pkl", "wb"))
        pickle.dump(vectorizer, open("vectorizer.pkl", "wb"))

        st.success("Model retrained successfully!")

    except Exception as e:
        st.error("No feedback data found or error occurred!")