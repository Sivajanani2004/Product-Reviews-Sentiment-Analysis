import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords")

# Load model and tokenizer
model = load_model("lstm_model.h5")
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Preprocessing
def clean_text(text):
    ps = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Predict sentiment
def predict_sentiment(text):
    text = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=100)
    prediction = model.predict(padded_sequence)[0][0]
    sentiment = "Positive" if prediction > 0.5 else "Negative"
    return sentiment, float(prediction)

# --- Streamlit Layout ---
st.set_page_config(page_title="Sentiment Analysis", page_icon="📊", layout="wide")

# Custom CSS for background, cards, hover, and gradient bars
st.markdown("""
<style>
/* Page background gradient */
.stApp {
    background: linear-gradient(to bottom right, #e0f7fa, #e8f5e9);
}

/* Textarea style */
textarea {
    background-color: #ffffffcc;
    border-radius: 10px;
    padding: 10px;
}

/* Buttons */
button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}

/* Review history cards */
.history-card {
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 10px;
    color: white;
    transition: transform 0.2s;
}
.history-card:hover {
    transform: scale(1.05);
}

/* Gradient confidence bar */
.progress-bar {
    height: 20px;
    border-radius: 10px;
    background: linear-gradient(to right, #4CAF50, #81C784);
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<h1 style='text-align:center; color:#4CAF50;'>🛒 Product Review Sentiment Dashboard</h1>
<p style='text-align:center; font-size:16px;'>Enter a product review to check its sentiment with confidence visualization.</p>
""", unsafe_allow_html=True)

# Initialize session state
if 'input' not in st.session_state:
    st.session_state['input'] = ""
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Sidebar example reviews
st.sidebar.header("Try Example Reviews")
examples = [
    "This product is amazing, I love it!",
    "Waste of money. Very disappointed.",
    "Good quality and fast delivery.",
    "Not satisfied with the product."
]
for ex in examples:
    if st.sidebar.button(ex):
        st.session_state['input'] = ex

# Main input
user_input = st.text_area("Type your review here:", value=st.session_state['input'], height=150)

col1, col2 = st.columns([2, 1])

with col1:
    if st.button("Analyze Review"):
        if user_input.strip() != "":
            sentiment, confidence = predict_sentiment(user_input)
            
            # Save to history
            st.session_state['history'].append({
                "review": user_input,
                "sentiment": sentiment,
                "confidence": confidence
            })
            
            # Sentiment display
            if sentiment == "Positive":
                emoji = "😊"
                color = "#4CAF50"
                progress = confidence
            else:
                emoji = "😡"
                color = "#F44336"
                progress = 1 - confidence
            
            # Animated emoji size based on confidence
            emoji_size = 50 + confidence * 50  # size 50-100 px

            st.markdown(f"""
            <div style='background-color:{color}; padding:20px; border-radius:15px; color:white; text-align:center;'>
                <h2 style='font-size:{emoji_size}px;'>{emoji}</h2>
                <h2>{sentiment} Review</h2>
                <p style='font-size:18px;'>Confidence: {confidence:.2f}</p>
                <div class="progress-bar" style="width:{progress*100}%;"></div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Please enter a review!")

with col2:
    st.markdown("### 📝 Review History")
    if st.session_state['history']:
        for item in reversed(st.session_state['history']):
            color = "#4CAF50" if item['sentiment']=="Positive" else "#F44336"
            st.markdown(f"""
            <div class="history-card" style='background-color:{color};'>
                {item['review']}<br>
                <b>Sentiment:</b> {item['sentiment']} | <b>Confidence:</b> {item['confidence']:.2f}
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No reviews analyzed yet.")

# Reset button
if st.button("Reset History"):
    st.session_state['history'] = []
    st.session_state['input'] = ""
