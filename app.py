import streamlit as st
import pickle
import numpy as np

# ==============================
# Load trained model & vectorizer
# ==============================
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

with open('models/xgboost_toxic_classifier.pkl', 'rb') as f:
    model = pickle.load(f)

# ==============================
# Prediction function
# ==============================
def predict_toxic_with_confidence(text, vectorizer, model):
    vectorized = vectorizer.transform([text])
    probabilities = model.predict_proba(vectorized)[0]

    normal_prob = float(probabilities[0])   # ⬅ convert to python float
    toxic_prob = float(probabilities[1])    # ⬅ convert to python float

    is_toxic = toxic_prob > 0.5
    confidence = toxic_prob if is_toxic else normal_prob

    # Confidence level
    if confidence > 0.85:
        confidence_level = "HIGH"
    elif confidence > 0.6:
        confidence_level = "MEDIUM"
    else:
        confidence_level = "LOW"

    return {
        "is_toxic": is_toxic,
        "normal_probability": normal_prob,
        "toxic_probability": toxic_prob,
        "confidence": confidence,
        "confidence_level": confidence_level
    }

# ==============================
# Streamlit UI
# ==============================

st.set_page_config(page_title="Toxic Comment Classifier", page_icon="💬", layout="centered")

st.title("🧪 Toxic Comment Classifier Demo")
st.write("Nhập 1 câu bất kỳ để mô hình dự đoán mức độ **toxic**.")

# User input
user_text = st.text_area("Nhập câu để dự đoán:", height=150)

if st.button("🔍 Dự đoán"):
    if not user_text.strip():
        st.warning("Vui lòng nhập nội dung.")
    else:
        result = predict_toxic_with_confidence(user_text, vectorizer, model)

        label = "🔴 TOXIC" if result["is_toxic"] else "🟢 NORMAL"
        st.subheader(f"Kết quả: {label}")

        # Probabilities
        st.write(f"**Toxic Probability:** `{result['toxic_probability']:.2%}`")
        st.write(f"**Normal Probability:** `{result['normal_probability']:.2%}`")
        st.write(f"**Confidence:** `{result['confidence']:.2%}` — *{result['confidence_level']}*")

        # Progress bars (MUST be python float)
        st.write("### 🔥 Toxic Probability")
        st.progress(float(result['toxic_probability']))

        st.write("### 🟢 Normal Probability")
        st.progress(float(result['normal_probability']))

# Footer
st.markdown("---")
st.caption("Built with Streamlit — Toxic Comment Classifier Demo")
