import streamlit as st

from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load the trained LSTM model
model = load_model('bidirectional_lstm_model.h5')  # Replace with the path to your .h5 model


# Function to preprocess the input text
def preprocess_text(review, tokenizer, max_length):
    sequences = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequences, maxlen=max_length, padding='post')
    return padded_sequence

# Streamlit app interface
st.title("Movie Review Sentiment Analysis")
st.write("This app uses an LSTM model to predict the sentiment of a movie review.")

# Input from the user
user_input = st.text_area("Enter a movie review:")

# If user submits a review
if st.button("Predict Sentiment"):
    if user_input:
        # Preprocess the user input
        max_length = 200  # Replace with the max_length used during training
        preprocessed_input = preprocess_text(user_input, max_length)
        
        # Predict the sentiment
        prediction = model.predict(preprocessed_input)
        sentiment = 'Positive' if prediction[0] >= 0.5 else 'Negative'
        
        # Display the result
        st.write(f"Sentiment: **{sentiment}**")
        st.write(f"Confidence Score: **{prediction[0]:.2f}**")
    else:
        st.write("Please enter a review to analyze.")
