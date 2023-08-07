from fastapi import FastAPI, HTTPException, Form

# Import your required libraries and modules
import pickle
import numpy as np

# Load pre-trained data and models using pickle
popular_df = pickle.load(open('popular.pkl', 'rb'))
pt = pickle.load(open('pt.pkl', 'rb'))
books = pickle.load(open('books.pkl', 'rb'))
similarity_scores = pickle.load(open('similarity_scores.pkl', 'rb'))

# Create a FastAPI application
app = FastAPI()

# Define a route '/recommend_books' for handling POST requests
@app.post('/recommend_books')
# Note: The `Form(...)` function is used to declare the 'book' parameter as a form field for handling POST requests.
# The ellipsis (...) specifies that the field is required.
# It is assumed that the 'url' for loading pre-trained data and models is correctly provided as in the previous code.

async def recommend(book: str = Form(...)):
    try:
        # Check if the book name is provided; if not, raise HTTP 400 error (Bad Request)
        if not book:
            raise HTTPException(status_code=400, detail='No book name provided.')

        # Find the index of the input book in the pivot table ('pt')
        index = np.where(pt.index == book)[0][0]

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
        return {'Recommended Books': data}

    # Handle exceptions and return an HTTP 500 error (Internal Server Error)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error occurred: {str(e)}')


# uvicorn main:app --reload


