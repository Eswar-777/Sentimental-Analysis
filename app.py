# app.py
import pandas as pd
import re
import nltk
from flask import Flask, request, render_template
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize Flask app
app = Flask(__name__)

# Download VADER lexicon if not already downloaded
nltk.download('vader_lexicon')

# Load VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

# Load the reviews dataset
data = pd.read_csv('reviews.csv')

# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^A-Za-z\s]', '', text)
    return text.lower()

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Sentiment analysis route
@app.route('/analyze', methods=['POST'])
def analyze():
    review = request.form['review']
    cleaned_review = preprocess_text(review)
    score = sid.polarity_scores(cleaned_review)
    sentiment = 'positive' if score['compound'] > 0.05 else 'negative' if score['compound'] < -0.05 else 'neutral'
    return {'sentiment': sentiment, 'score': score}

if __name__ == '__main__':
    app.run(debug=True)
