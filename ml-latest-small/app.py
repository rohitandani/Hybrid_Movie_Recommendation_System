from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import random
import dash
from dash import dcc, html
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64
import os
import requests

# Initialize Flask app with the specified template folder
app = Flask(__name__, template_folder=r'C:\Users\rohit\Downloads\Project\m1-latest-small\templates')
#app = Flask(__name__, template_folder=r'https://raw.githubusercontent.com/rohitandani/Hybrid_Movie_Recommendation_System/main/ml-latest-small/')
# Load datasets from provided URLs
ratings = pd.read_csv('https://raw.githubusercontent.com/rohitandani/Hybrid_Movie_Recommendation_System/main/ml-latest-small/ml-latest-small/ratings.csv')
movies = pd.read_csv('https://raw.githubusercontent.com/rohitandani/Hybrid_Movie_Recommendation_System/main/ml-latest-small/ml-latest-small/movies.csv')
tags = pd.read_csv('https://raw.githubusercontent.com/rohitandani/Hybrid_Movie_Recommendation_System/main/ml-latest-small/ml-latest-small/tags.csv')

# Merge tags with movies
movie_tags = tags.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)).reset_index()
movies_with_tags = pd.merge(movies, movie_tags, on='movieId', how='left').fillna('')

# Create the user-item matrix
user_item_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)

# Calculate cosine similarity between users
user_similarity_df = cosine_similarity(user_item_matrix)
user_similarity_df = pd.DataFrame(user_similarity_df, index=user_item_matrix.index, columns=user_item_matrix.index)

# Calculate cosine similarity between items
item_similarity_df = cosine_similarity(user_item_matrix.T)
item_similarity_df = pd.DataFrame(item_similarity_df, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# Create a combined feature (genres + tags) for the content-based filtering
movies_with_tags['combined_features'] = movies_with_tags['genres'] + ' ' + movies_with_tags['tag']

# Create a TF-IDF matrix for the combined features
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(movies_with_tags['combined_features'])

# Calculate cosine similarity between movies
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# Function to generate and encode heatmap images
def generate_heatmap(matrix, title, x_label, y_label):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='coolwarm', annot=False, fmt='.2f')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()
    return image_base64

# Recommendation functions
def get_user_based_recommendations(user_id, num_recommendations):
    user_ratings = user_item_matrix.loc[user_id]
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    recommendations = {}
    
    for similar_user, similarity in similar_users.items():
        if similar_user != user_id:
            similar_user_ratings = user_item_matrix.loc[similar_user]
            for movie_id, rating in similar_user_ratings.items():
                if user_ratings[movie_id] == 0:  # If the user has not rated this movie
                    if movie_id not in recommendations:
                        recommendations[movie_id] = rating * similarity
                    else:
                        recommendations[movie_id] += rating * similarity
    
    sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
    random.shuffle(sorted_recommendations)
    
    return sorted_recommendations[:num_recommendations]

def get_item_based_recommendations(user_id, num_recommendations):
    user_ratings = user_item_matrix.loc[user_id]
    recommendations = {}
    
    for movie_id, rating in user_ratings.items():
        if rating > 0:
            similar_items = item_similarity_df[movie_id].sort_values(ascending=False)
            for similar_movie_id, similarity in similar_items.items():
                if user_ratings[similar_movie_id] == 0:  # If the user has not rated this movie
                    if similar_movie_id not in recommendations:
                        recommendations[similar_movie_id] = rating * similarity
                    else:
                        recommendations[similar_movie_id] += rating * similarity
    return sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:num_recommendations]

def get_content_based_recommendations(movie_id, num_recommendations):
    movie_index = movies_with_tags[movies_with_tags['movieId'] == movie_id].index[0]
    similarity_scores = list(enumerate(cosine_sim[movie_index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[1:num_recommendations+1]
    movie_indices = [i[0] for i in similarity_scores]
    return movies_with_tags.iloc[movie_indices]

def hybrid_recommendations(user_id, movie_id, num_recommendations):
    user_based_recs = get_user_based_recommendations(user_id, num_recommendations)
    item_based_recs = get_item_based_recommendations(user_id, num_recommendations)
    content_based_recs = get_content_based_recommendations(movie_id, num_recommendations)
    
    user_based_movie_ids = [rec[0] for rec in user_based_recs]
    item_based_movie_ids = [rec[0] for rec in item_based_recs]
    content_based_movie_ids = content_based_recs['movieId'].tolist()
    
    # Combine all the movie IDs into a set to remove duplicates, then back to a list
    combined_movie_ids = list(set(user_based_movie_ids + item_based_movie_ids + content_based_movie_ids))
    
    # Sort combined recommendations and return only the top recommendations
    top_recommendations = combined_movie_ids[:num_recommendations]
    
    return top_recommendations

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form['user_id'])
        movie_id = int(request.form['movie_id'])
        num_recommendations = int(request.form['num_recommendations'])

        recommendations = hybrid_recommendations(user_id, movie_id, num_recommendations)
        recommended_movies = movies_with_tags[movies_with_tags['movieId'].isin(recommendations)][['title', 'genres', 'tag']]
        user_item_heatmap = generate_heatmap(user_similarity_df, 'User Similarity Heatmap', 'User ID', 'User ID')
        item_item_heatmap = generate_heatmap(item_similarity_df, 'Item Similarity Heatmap', 'Movie ID', 'Movie ID')

        return render_template('recommendations.html', tables=[recommended_movies.to_html(classes='data')], titles=recommended_movies.columns.values, user_item_heatmap=user_item_heatmap, item_item_heatmap=item_item_heatmap)
    except Exception as e:
        return f"An error occurred: {e}"

# Create Dash app within Flask app
dash_app = dash.Dash(__name__, server=app, url_base_pathname='/dash/')

# Sample data for visualization
sample_ratings = ratings.sample(n=1000)

# Plotly express figure for ratings histogram
ratings_fig = px.histogram(sample_ratings, x='rating', nbins=10, title='Distribution of Ratings')

# Plotly express figures for heatmaps
user_sim_fig = px.imshow(user_similarity_df, color_continuous_scale='RdBu_r', aspect='auto', title='User Similarity Heatmap')
item_sim_fig = px.imshow(item_similarity_df, color_continuous_scale='RdBu_r', aspect='auto', title='Item Similarity Heatmap')

# Layout of the Dash app
dash_app.layout = html.Div(children=[
    html.H1(children='Interactive Visualization Dashboard'),
    
    dcc.Graph(
        id='ratings-histogram',
        figure=ratings_fig
    ),

    html.H2(children='User Similarity Heatmap'),
    dcc.Graph(
        id='user-sim-heatmap',
        figure=user_sim_fig
    ),

    html.H2(children='Item Similarity Heatmap'),
    dcc.Graph(
        id='item-sim-heatmap',
        figure=item_sim_fig
    ),
])

# Run the Flask app
if __name__ == '__main__':
    try:
        app.run(debug=True)
    except SystemExit:
        pass  # Ignore SystemExit to avoid stopping the notebook