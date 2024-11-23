import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import string
import nltk
import spacy
import re
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import joblib

# Load data
d1 = pd.read_csv(r"movies.csv")

Data=d1
# Drop unnecessary columns
Data.drop(columns={"movieId"}, inplace=True)
Data.drop_duplicates(inplace=True)

# Slice the dataset for performance
Dataa = Data.iloc[40000:]

# Clean the titles
Dataa["title"] = [re.sub("[^a-zA-z]", " ", i) for i in Dataa["title"]]
Dataa["title"] = [i.strip() for i in Dataa["title"]]

# Function to clean genres
def clean(genres):
    genres = genres.lower()
    genres = re.sub(r'\|', ' ', genres)
    genres = genres.split()
    return genres

# Apply genre cleaning
Dataa["Genres1"] = Dataa["genres"].apply(lambda X: clean(X))
Dataa.drop(columns={"genres"}, inplace=True)
Dataa["Genres1"] = Dataa['Genres1'].apply(lambda x: ' '.join(x))

# Reset index for consistency
Dataa = Dataa.reset_index(drop=True)

# Vectorizing the genres column
cv = CountVectorizer(max_features=5000)
vectors = cv.fit_transform(Dataa["Genres1"]).toarray()

# Calculate similarity matrix
similarity = cosine_similarity(vectors)

# Movie recommendation function
def recommend(movie_name):
    try:
        # Get the index of the movie
        movie_index = Dataa[Dataa["title"] == movie_name].index[0]
        
        # Get the similarity scores for that movie
        distances = similarity[movie_index]
        
        # Sort movies based on similarity
        movie_list = sorted(list(enumerate(distances)), reverse=True, key=lambda X: X[1])[1:6]  # get top 5 recommendations
        
        # Collect recommended movie titles
        recommended_movies = []
        for i in movie_list:
            recommended_movies.append(Dataa.iloc[i[0]].title)
        
        # Return the list of recommended movies
        return recommended_movies
    
    except IndexError:
        # In case the movie name is not found
        return ["Movie not found in the dataset. Please check the spelling or try another movie."]
    

# Streamlit interface
st.title("Welcome to Movie Recommender")

# Input field for movie name
movie_name = st.text_input("Enter Movie name")

# Button to get recommendations
button = st.button("Click For Recommendation")

if button:
    # Get recommendations and display them
    recommended_movies = recommend(movie_name)
    
    # Display recommendations in subheader
    if isinstance(recommended_movies, list):
        for movie in recommended_movies:
            st.subheader(movie)
    else:
        st.subheader("your movies is not in Database")