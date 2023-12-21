import numpy as np
import pandas as pd
import re

import tkinter as tk
from tkinter import Text, END
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import tensorflow_datasets as tfds

# Load IMDb dataset
imdb = pd.read_csv("IMDBDataset.csv")

# Clean HTML tags
def clean_html(raw_html):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', raw_html)
    return cleantext

imdb['review'] = imdb['review'].apply(clean_html)

# Prepare Tokenizer
Tok_data = Tokenizer(oov_token='<OOV>')
Tok_data.fit_on_texts(imdb['review'])
vocab_size = len(Tok_data.word_index) + 1

# Pad text
max_length = 100
encoded_reviews = Tok_data.texts_to_sequences(imdb['review'])
padded_reviews = pad_sequences(encoded_reviews, maxlen=max_length, padding='post')

# Load LSTM model
model_lstm = load_model("your_lstm_model.h5")

# Tkinter interface
def analyze_sentiment():
    input_text = input_text_entry.get("1.0", tk.END)
    input_text = clean_html(input_text)
    tokens = Tok_data.texts_to_sequences([input_text])
    padded_tokens = pad_sequences(tokens, maxlen=max_length, padding='post')
    prediction = model_lstm.predict(padded_tokens)
    result_label.config(text=f"Sentiment: {'Positive' if prediction[0] > 0.5 else 'Negative'}")

# GUI setup
root = tk.Tk()
root.title("Sentiment Analysis")

# Text entry for user input
input_text_entry = Text(root, wrap="word", width=50, height=10)
input_text_entry.pack(padx=10, pady=10)

# Button to trigger sentiment analysis
analyze_button = tk.Button(root, text="Analyze Sentiment", command=analyze_sentiment)
analyze_button.pack(pady=10)

# Label to display sentiment result
result_label = tk.Label(root, text="")
result_label.pack()

# Run the Tkinter event loop
root.mainloop()
