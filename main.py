import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import tkinter as tk
from PIL import Image, ImageTk

def train_sentiment_model():
    youtube = pd.read_csv("YOUTUBE.csv")

    sentiment_mapping = {'negative': 0, 'positive': 1}
    youtube['sentiment'] = youtube['sentiment'].map(sentiment_mapping)

    X = youtube['comment']
    y = youtube['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vectorizer = TfidfVectorizer()
    X_train_vect = vectorizer.fit_transform(X_train)
    X_test_vect = vectorizer.transform(X_test)

    sentiment_model = LogisticRegression()
    sentiment_model.fit(X_train_vect, y_train)

    return sentiment_model, vectorizer

def check_sentiment():
    text = entry.get()
    text_vectorized = vectorizer.transform([text])
    
    prediction = sentiment_model.predict(text_vectorized)
    
    sentiment_mapping_reverse = {0: 'Negative', 1: 'Positive'}
    result_label.config(text=f"This comment is: {sentiment_mapping_reverse[prediction[0]]}", fg="red" if prediction[0] == 0 else "green")

sentiment_model, vectorizer = train_sentiment_model()





root = tk.Tk()
root.title("YOUTUBE Comments Sentiment Analysis")

window_width = 800
window_height = 500
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
x_coordinate = int((screen_width / 2) - (window_width / 2))
y_coordinate = int((screen_height / 2) - (window_height / 2))
root.geometry(f"{window_width}x{window_height}+{x_coordinate}+{y_coordinate}")

bg_image = Image.open("back.png")
bg_image = ImageTk.PhotoImage(bg_image)
bg_label = tk.Label(root, image=bg_image)
bg_label.place(relwidth=1, relheight=1)

entry = tk.Entry(root, width=60, font=("Arial", 14))
entry.place(relx=0.5, rely=0.5, anchor="center")

button_image = Image.open("button.png")
button_image = button_image.resize((button_image.size[0] // 2, button_image.size[1] // 2))
button_image = ImageTk.PhotoImage(button_image)
check_button = tk.Button(root, image=button_image, command=check_sentiment, bd=0)
check_button.place(relx=0.5, rely=0.7, anchor="center")

result_label = tk.Label(root, text="", font=("Arial", 16))
result_label.place(relx=0.5, rely=0.9, anchor="center")

root.mainloop()
