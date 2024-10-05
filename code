# Book-Recommendation
Book recommedation using AI

The code was written with help of AI and i have personally made changes where i think the code was too long and lastly there are no error this 



import pandas as pd

book_df = pd.read_csv('Books.csv')
ratings_df = pd.read_csv('Ratings.csv').sample(40000)
user_df = pd.read_csv('Users.csv')
user_rating_df = ratings_df.merge(user_df, left_on = 'User-ID', right_on = 'User-ID')

user_rating_df.head()

book_user_rating = book_df.merge(user_rating_df, left_on = 'ISBN',right_on = 'ISBN')
book_user_rating = book_user_rating[['ISBN', 'Book-Title', 'Book-Author', 'User-ID', 'Book-Rating']]
book_user_rating.reset_index(drop=True, inplace = True)

book_user_rating.head()

d ={}
for i,j in enumerate(book_user_rating.ISBN.unique()):
    d[j] =i
book_user_rating['unique_id_book'] = book_user_rating['ISBN'].map(d)
book_user_rating.head(50)
users_books_pivot_matrix_df = book_user_rating.pivot(index='User-ID', 
                                                          columns='unique_id_book', 
                                                          values='Book-Rating').fillna(0)
                                                          
users_books_pivot_matrix_df = users_books_pivot_matrix_df.values
users_books_pivot_matrix_df
from scipy.sparse.linalg import svds

NUMBER_OF_FACTORS_MF = 15

#Performs matrix factorization of the original user item matrix
U, sigma, Vt = svds(users_books_pivot_matrix_df, k = NUMBER_OF_FACTORS_MF)

import numpy as np

sigma = np.diag(sigma)
sigma.shape

all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) 
all_user_predicted_ratings


def compute_cosine_similarity(data):
    """Compute cosine similarity matrix for all books."""
    # Calculate the magnitude of each book's feature vector
    magnitude = np.sqrt(np.einsum('ij, ij -> i', data, data))
    
    # Normalize the data by dividing each row by its magnitude
    normalized_data = data / magnitude[:, np.newaxis]
    
    # Compute the cosine similarity matrix
    cosine_similarity_matrix = np.dot(normalized_data, normalized_data.T)
    
    return cosine_similarity_matrix

def top_cosine_similarity(cosine_similarity_matrix, book_id, top_n=10):
    """Get top N similar books based on cosine similarity."""
    # Get the similarity scores for the specified book
    similarity_scores = cosine_similarity_matrix[book_id]
    
    # Get the indices of the top N similar books (excluding itself)
    sort_indexes = np.argsort(-similarity_scores)
    
    return sort_indexes[sort_indexes != book_id][:top_n]

def similar_books(book_user_rating, book_id, top_indexes):
    """Print recommended books based on similar books."""
    print('Recommendations for {0}: \n'.format(
        book_user_rating[book_user_rating.unique_id_book == book_id]['Book-Title'].values[0]))
    
    for id in top_indexes:
        print(book_user_rating[book_user_rating.unique_id_book == id]['Book-Title'].values[0])

     k = 50
movie_id =25954  
top_n = 3
sliced = Vt.T[:, :k] # representative data

similar_books(book_user_rating, 25954, top_cosine_similarity(sliced, movie_id, top_n))   
