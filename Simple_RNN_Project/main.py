from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences
import streamlit as st


word_index = imdb.get_word_index()
reverse_word_index = {value:key for key,value in word_index.items()}

model = load_model('simple_rnn_model.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3,'?') for i in encoded_review])

def preprocessed_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = pad_sequences([encoded_review],maxlen = 500)
    return padded_review


st.title('Sentiment analysis of Movie Review')
st.write('Enter the movie review')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocessed_text(user_input)
    prediction = model.predict(preprocessed_input)
    
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction: {prediction[0][0]}')
else:
    st.write('Please enter a movie review')