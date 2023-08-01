from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# Load the pre-trained Keras model and tokenizer
model = load_model('next_words.h5')
tokenizer = pickle.load(open('token.pkl', 'rb'))

# Function to predict the next word given a sequence of three words
def Predict_Next_Words(model, tokenizer, text):
    # Convert the input text to a sequence of numbers using the tokenizer
    sequence = tokenizer.texts_to_sequences([text])
    sequence = np.array(sequence)

    # Make predictions using the model
    preds = np.argmax(model.predict(sequence))

    # Convert the predicted number back to the corresponding word using the tokenizer
    predicted_word = ""
    for key, value in tokenizer.word_index.items():
        if value == preds:
            predicted_word = key
            break

    return predicted_word

# Define the route to handle the API request
@app.route('/Nextword', methods=['POST'])
def nextword():
    try:
        # Get the input data from the request form with the key "text"
        text_ = request.form['text']

        # Split the input text into individual words and extract the last three words
        text_ = text_.split(" ")
        text_ = text_[-3:]  # Ensure that text_ contains at least three words

        # If text_ contains less than three words, pad it with empty strings
        text_ = [''] * (3 - len(text_)) + text_

        # Convert the list of words back to a single string with spaces between them
        text_ = " ".join(text_)

        # Predict the next word using the model and the pre-processed input text
        nextword = Predict_Next_Words(model, tokenizer, text_)

        # Return the predicted next word as a JSON response
        return jsonify({'Your next word is': nextword}), 200

    except Exception as e:
        # If there is an error, return an error message as a JSON response
        return jsonify({'error': f'Error occurred: {str(e)}'}), 500

# Start the Flask application
if __name__ == '__main__':
    app.run(debug=True)
