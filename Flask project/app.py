from flask import Flask, request, render_template
import joblib
import re

app = Flask(__name__)

# Load the trained model
clf = joblib.load('sentiment_model.h5')
# Load the fitted vectorizer
vectorizer = joblib.load('fitted_vectorizer.h5')

# Preprocess the input text
def preprocess_text(text):
    text = re.sub('[^\w\s]', '', text) # Remove punctuations
    text = text.lower() # Convert to lowercase
    return text

def predict_sentiment(text):
    # preprocess text
    preprocessed_text = preprocess_text(text)
    # vectorize text
    vectorized_text = vectorizer.transform([preprocessed_text])
    # predict sentiment
    sentiment = clf.predict(vectorized_text)[0]
    # return sentiment label as string
    return "Positive" if sentiment == 1 else "Negative"




# Define the home page
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input text from the user
    text = request.form['text']
    
    sentiment = predict_sentiment(text)
    css_file = 'positive.css' if sentiment == 'Positive' else 'negative.css'

    # Return the merged template with the predicted sentiment
    return render_template('index.html', sentiment=sentiment, css_file=css_file)

if __name__ == '__main__':
    app.run(debug=True)