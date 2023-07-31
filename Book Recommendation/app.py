# Import required libraries and modules
from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load pre-trained data and models using pickle
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))

# Create a Flask application
app = Flask(__name__)

# Define a route '/recommend_books' for handling POST requests
@app.route('/recommend_books', methods=['POST'])
def recommend():
    try:
        # Get the book name from the form data submitted in the POST request
        book_name = request.form.get('book')

        # Check if the book name is provided; if not, return an error response with status code 400 (Bad Request)
        if not book_name:
            return jsonify({'error': 'No book name in the form data.'}), 400

        # Find the index of the input book in the pivot table ('pt')
        index = np.where(pt.index == book_name)[0][0]

        # Calculate the most similar items (books) based on similarity scores
        similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

        # Prepare the data for recommended books, including titles, authors, and image URLs
        data = []
        for i in similar_items:
            item = []
            temp_df = books[books['Book-Title'] == pt.index[i[0]]]
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
            item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))

            data.append(item)

        # Return a JSON response containing the recommended books
        return jsonify({'Recommended Books': data}), 200

    # Handle exceptions and return an error response with status code 500 (Internal Server Error)
    except Exception as e:
        raise e
        return jsonify({'error': f'Error occurred: {str(e)}'}), 500

# Run the Flask application in debug mode if the script is directly executed
if __name__ == '__main__':
    app.run(debug=True)
