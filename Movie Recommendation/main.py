import numpy as np
import pandas as pd
import ast

# Read movie data from CSV files: 'tmdb_5000_movies.csv' and 'tmdb_5000_credits.csv'
movies = pd.read_csv('tmdb_5000_movies.csv')
creditss = pd.read_csv('tmdb_5000_credits.csv')

# Merge movie and credits data on 'title'
movies = movies.merge(creditss, on='title')

# Keep only relevant columns for the recommendation system
movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# Check for missing values (NaN) in the movie data
null = movies.isnull().sum()

# Drop rows with missing data
movies.dropna(inplace=True)

# Function to convert stringified lists to actual lists
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return L

# Apply 'convert' function to 'genres' and 'keywords' columns
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Function to extract top 3 cast members from the cast list
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter += 1
    return L

# Apply 'convert3' function to 'cast' column
movies['cast'] = movies['cast'].apply(convert3)

# Function to fetch director(s) from the crew list
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L

# Apply 'fetch_director' function to 'crew' column
movies['crew'] = movies['crew'].apply(fetch_director)

# Function to remove spaces from strings in a list
def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ", ""))
    return L1

# Apply 'collapse' function to 'cast', 'crew', 'genres', and 'keywords' columns
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)

# Split words in 'overview', 'genres', 'keywords', 'cast', and 'crew' columns into individual tokens
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Combine 'overview', 'genres', 'keywords', 'cast', and 'crew' into a new column 'tags'
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

# Drop unnecessary columns ('overview', 'genres', 'keywords', 'cast', 'crew')
new = movies.drop(columns=['overview', 'genres', 'keywords', 'cast', 'crew'])

# Convert the list of tags into space-separated strings
new['tags'] = new['tags'].apply(lambda x: " ".join(x))

# Vectorize 'tags' column using CountVectorizer to create a matrix of token counts
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')
vector = cv.fit_transform(new['tags']).toarray()

# Calculate cosine similarity between movies based on the tag counts
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)

# Function to recommend movies based on movie name
def recommend(movie):
    # Find the index of the input movie in the DataFrame
    index = new[new['title'] == movie].index[0]
    
    # Sort movies based on similarity with the input movie
    distances = sorted(list(enumerate(similarity[index])), reverse=True, key=lambda x: x[1])

    # Print the top 5 recommended movie titles
    for i in distances[1:6]:
        print(new.iloc[i[0]].title)

# Save preprocessed movie data and similarity matrix using pickle
import pickle
pickle.dump(new, open('movie_list.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))
