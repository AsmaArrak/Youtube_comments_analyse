# YouTube Comments Sentiment Analysis 🎥📝

This project performs sentiment analysis on YouTube comments using a logistic regression model. It includes a graphical user interface (GUI) built with Tkinter, allowing users to input a comment and receive a sentiment analysis result.

## Requirements 🛠️
- Python 3.x
- Required Python packages: `numpy`, `pandas`, `scikit-learn`, `tkinter`, `PIL`

## Installation 🚀
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repository.git
   cd your-repository

## Usage 📊
Ensure that the necessary CSV file (YOUTUBE.csv) is present in the project directory.

Run the script:

python your_script_name.py
Replace your_script_name.py with the actual name of your script.

The GUI window will appear with an input field and a button.

Enter a YouTube comment in the input field.

Click the button to analyze the sentiment of the entered comment.

The result will be displayed on the GUI, indicating whether the sentiment is positive or negative.

## Models Used 🤖
The sentiment analysis is performed using the following models:

Logistic Regression: The main sentiment classification model trained on the provided YouTube comments dataset.

TfidfVectorizer: Used to convert text data into numerical features for the logistic regression model.

## Files 📂

main.py: The main Python script containing the sentiment analysis logic and GUI implementation.
YOUTUBE.csv: CSV file containing YouTube comments and their sentiments.
back.png: Background image for the GUI.
button.png: Image used for the analysis button
