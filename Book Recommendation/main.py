import pandas as pd
import numpy as np

# Read CSV files into DataFrames
books = pd.read_csv("Books.csv")
ratings = pd.read_csv("Ratings.csv")
users = pd.read_csv("Users.csv")

# Check for null values in each DataFrame
booksnull = books.isnull().sum()
ratingsnull = ratings.isnull().sum()
usersnull = users.isnull().sum()

# Merge 'books' and 'ratings' DataFrames on 'ISBN' to get a DataFrame with ratings and book titles
ratingwithtitle = books.merge(ratings, on="ISBN")

# Group by 'Book-Title' to get the number of ratings for each book
num_rating_df = ratingwithtitle.groupby('Book-Title').count()['Book-Rating'].reset_index()
num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

# Group by 'Book-Title' to calculate the average rating for each book
avg_rating_df = ratingwithtitle.groupby('Book-Title')['Book-Rating'].mean().reset_index()
avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

# Merge 'num_rating_df' and 'avg_rating_df' to create a DataFrame with the number of ratings and average rating for each book
popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')

# Filter books with at least 250 ratings and sort them by average rating in descending order
popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False)

# Merge 'popular_df' with the 'books' DataFrame to get additional book information for popular books
popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']]

# Create a pivot table 'pt' with 'User-ID' as columns, 'Book-Title' as rows, and 'Book-Rating' as values
x = ratingwithtitle.groupby('User-ID').count()['Book-Rating'] > 200
Rated_users = x[x].index
filtered_rating = ratingwithtitle[ratingwithtitle['User-ID'].isin(Rated_users)]
y = filtered_rating.groupby('Book-Title').count()['Book-Rating'] >= 50
famous_books = y[y].index
final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
pt.fillna(0, inplace=True)

# Calculate similarity scores between books using cosine similarity
from sklearn.metrics.pairwise import cosine_similarity
similarity_scores = cosine_similarity(pt)

# Function to recommend books based on similarity scores
def recommend(book_name):
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]
    
    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    
    return data

# Save data and models using pickle
import pickle
pickle.dump(popular_df, open('popular.pkl', 'wb'))
pickle.dump(pt, open('pt.pkl', 'wb'))
pickle.dump(books, open('books.pkl', 'wb'))
pickle.dump(similarity_scores, open('similarity_scores.pkl', 'wb'))
