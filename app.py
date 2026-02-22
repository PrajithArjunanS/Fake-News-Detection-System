import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords


st.set_page_config(
    page_title="Fake News Detection System",
    layout="centered"
)


st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    height: 100%;
}

.main {
    display: flex;
    flex-direction: column;
    min-height: 100vh;
}

.block-container {
    flex: 1;
    padding-top: 2rem;
}

.stTextArea textarea {
    border-radius: 12px;
}

.stButton>button {
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    font-weight: 600;
}

.stProgress > div > div > div > div {
    border-radius: 10px;
}

footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


stop_words = set(stopwords.words("english"))


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

st.markdown("""
<h1 style='text-align:center;'>Fake News Detection System</h1>
<p style='text-align:center; color:grey; font-size:16px;'>
AI-Based News Credibility Classification Tool
</p>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


st.subheader("Enter News Article Text")
user_input = st.text_area("", height=200)


if st.button("Analyze"):
    if user_input:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])

        prediction = model.predict(vec)[0]
        probability = model.predict_proba(vec)[0]

        if prediction == 1:
            st.markdown("""
            <div style="
            padding:18px;
            border-radius:14px;
            background-color:#e6f4ea;
            color:#1b5e20;
            text-align:center;
            font-size:20px;
            font-weight:600;
            border:1px solid #c8e6c9;">
            Prediction: REAL NEWS
            </div>
            """, unsafe_allow_html=True)
            confidence = probability[1]
        else:
            st.markdown("""
            <div style="
            padding:18px;
            border-radius:14px;
            background-color:#fdecea;
            color:#b71c1c;
            text-align:center;
            font-size:20px;
            font-weight:600;
            border:1px solid #f5c6cb;">
            Prediction: FAKE NEWS
            </div>
            """, unsafe_allow_html=True)
            confidence = probability[0]

        st.markdown("<br>", unsafe_allow_html=True)
        st.write(f"Confidence: {round(confidence * 100, 2)}%")
        st.progress(float(confidence))

    else:
        st.warning("Please enter some text.")


st.markdown("<div style='flex-grow:1;'></div>", unsafe_allow_html=True)

st.markdown("""
<hr>
<div style='text-align:center; font-size:14px; color:grey;'>
Developed by 
<a href="https://www.linkedin.com/in/sparjunan" target="_blank" 
style="color:#4da6ff; text-decoration:none; font-weight:600;">
Prajith Arjunan S
</a> 
| AI-Based Fake News Detection System
</div>
<div style='text-align:center; font-size:13px; color:grey; margin-top:8px;'>
Disclaimer: This system uses machine learning predictions and may not be 100% accurate.
</div>
""", unsafe_allow_html=True)