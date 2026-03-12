import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl","rb"))
vectorizer = pickle.load(open("vectorizer.pkl","rb"))

st.title("📰 Fake News Detection AI")

st.write("Enter a news article below to check if it is Fake or Real.")

news_text = st.text_area("Enter News Text")

if st.button("Predict"):

    data = vectorizer.transform([news_text])
    prediction = model.predict(data)

    if prediction[0] == 0:
        st.error("🚨 Fake News")
    else:
        st.success("✅ Real News")