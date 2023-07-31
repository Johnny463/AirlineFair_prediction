from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load pre-trained movie list and similarity data using pickle
movies = pickle.load(open('movie_list.pkl', 'rb'))
similarity = pickle.load(open('similarity.pkl', 'rb'))

# Define a route '/rmovie' for handling POST requests
@app.route('/rmovie', methods=['POST'])
def recommended_movie():
    try:
        # Get the movie name from the form data submitted in the POST request
        movie_name = request.form['movie']

        # Check if the movie name is provided; if not, return an error response with status code 400 (Bad Request)
        if not movie_name:
            return jsonify({'error': 'No movie name in the form data.'}), 400

        # Find the index of the input movie in the 'movies' DataFrame
        index = movies[movies['title'] == movie_name].index[0]

        # Sort the movies based on similarity with the input movie
        distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

        # Prepare a list of recommended movie names
        recommended_movie_names = []
        for i in distances[1:6]:
            recommended_movie_names.append(movies.iloc[i[0]]['title'])

        # Return a JSON response containing the recommended movie names
        return jsonify({'Recommended Movies': recommended_movie_names}), 200

    # Handle exceptions and return an error response with status code 500 (Internal Server Error)
    except Exception as e:
        raise e
        return jsonify({'error': f'Error occurred: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)
