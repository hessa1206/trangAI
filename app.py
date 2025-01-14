import streamlit as st
import tensorflow as tf
import json
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import requests

# Hàm tải tệp từ Google Drive
def download_from_google_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    
    response = session.get(URL, params={'id': file_id}, stream=True)
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            response = session.get(URL, params={'id': file_id, 'confirm': value}, stream=True)
    
    with open(destination, "wb") as f:
        for chunk in response.iter_content(chunk_size=32768):
            if chunk:
                f.write(chunk)

# Hàm tải mô hình và tokenizer
@st.cache_resource
def load_model_and_tokenizer():
    # Tải mô hình từ Google Drive
    model_file_id = "1CylcNlu8istLXCnOuhLB2JgXMvGp3jYB"  # Thay bằng Google Drive file ID của bạn
    tokenizer_file_id = "1t9J4zvCIxmiFYNDmVjh75UcKtnD3ARJF"  # Thay bằng Google Drive file ID của tokenizer
    model_destination = "fake_news_model_detection_.h5"
    tokenizer_destination = "tokenizer.json"
    
    # Tải tệp nếu chưa tồn tại
    download_from_google_drive(model_file_id, model_destination)
    download_from_google_drive(tokenizer_file_id, tokenizer_destination)
    
    # Load mô hình và tokenizer
    model = tf.keras.models.load_model(model_destination)
    with open(tokenizer_destination, 'r') as f:
        tokenizer = tokenizer_from_json(json.load(f))
    return model, tokenizer

# Load model và tokenizer
model, tokenizer = load_model_and_tokenizer()

# Hàm tiền xử lý
def preprocess_text(text, tokenizer, max_len=300):
    sequences = tokenizer.texts_to_sequences([text])
    return pad_sequences(sequences, maxlen=max_len, padding='post')

# Hàm dự đoán
def predict_news(text, model, tokenizer):
    input_data = preprocess_text(text, tokenizer)
    prediction = model.predict(input_data)[0][0]
    return "Fake News" if prediction > 0.5 else "Real News"

# Streamlit UI
st.title("Fake News Detection")
st.write("Paste a news article below to check if it's real or fake.")

# Nhập văn bản
user_input = st.text_area("Enter your news article:")
if st.button("Check News"):
    if user_input.strip():
        result = predict_news(user_input, model, tokenizer)
        st.write(f"The article is likely: **{result}**")
    else:
        st.write("Please enter some text.")