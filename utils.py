import pickle
import re
import pandas as pd

def load_models():
    """Load trained model and vectorizer"""
    try:
        with open('models/tfidf_vectorizer.pkl', 'rb') as f:
            vectorizer = pickle.load(f)
        
        with open('models/xgboost_toxic_classifier.pkl', 'rb') as f:
            model = pickle.load(f)
        
        return vectorizer, model
    except FileNotFoundError as e:
        raise Exception(f"Model files not found: {e}")

def preprocess_text(text):
    """
    Preprocessing function - phải giống với lúc training
    """
    if pd.isna(text) or text == '':
        return ""
    
    text = str(text)
    
    # Xóa URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Xóa mentions
    text = re.sub(r'@\w+', '', text)
    
    # Xóa ký tự đặc biệt (giữ tiếng Việt)
    text = re.sub(r'[^\w\s\u00C0-\u1EF9,.!?]', ' ', text)
    
    # Chuẩn hóa khoảng trắng
    text = ' '.join(text.split())
    
    # Lowercase
    text = text.lower().strip()
    
    return text

def predict_toxic(text, vectorizer, model):
    """
    Predict toxic classification with confidence
    
    Returns:
    --------
    dict with prediction results
    """
    if not text or text.strip() == '':
        return {
            'error': 'Vui lòng nhập nội dung tin nhắn',
            'is_toxic': False,
            'label': 0,
            'confidence': 0.0,
            'confidence_level': 'N/A',
            'toxic_probability': 0.0,
            'normal_probability': 0.0
        }
    
    # Preprocess
    cleaned_text = preprocess_text(text)
    
    if cleaned_text == '':
        return {
            'error': 'Không thể xử lý nội dung này',
            'is_toxic': False,
            'label': 0,
            'confidence': 0.0,
            'confidence_level': 'N/A',
            'toxic_probability': 0.0,
            'normal_probability': 0.0
        }
    
    # Vectorize
    text_vec = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vec)[0]
    probabilities = model.predict_proba(text_vec)[0]
    
    # Confidence = probability of predicted class
    confidence = probabilities[prediction]
    
    # Confidence level
    if confidence < 0.5:
        conf_level = "Rất Thấp"
        conf_color = "🟡"
    elif confidence < 0.7:
        conf_level = "Thấp"
        conf_color = "🟠"
    elif confidence < 0.85:
        conf_level = "Trung Bình"
        conf_color = "🟠"
    elif confidence < 0.95:
        conf_level = "Cao"
        conf_color = "🔴"
    else:
        conf_level = "Rất Cao"
        conf_color = "🔴"
    
    return {
        'original_text': text,
        'cleaned_text': cleaned_text,
        'is_toxic': bool(prediction),
        'label': int(prediction),
        'confidence': float(confidence),
        'confidence_level': conf_level,
        'confidence_color': conf_color,
        'toxic_probability': float(probabilities[1]),
        'normal_probability': float(probabilities[0])
    }

def get_confidence_color(confidence):
    """Get color based on confidence level"""
    if confidence < 0.5:
        return "#FFA500"  # Orange
    elif confidence < 0.7:
        return "#FF6B6B"  # Light red
    elif confidence < 0.85:
        return "#EE5A6F"  # Red
    elif confidence < 0.95:
        return "#C92A2A"  # Dark red
    else:
        return "#8B0000"  # Very dark red