pip install pymongo
import pymongo
from pymongo import MongoClient
client=MongoClient("localhost",27017)
db=client['Movies']
movies=db["meta"]


import pandas as pd
rate=db['Ratings_Small']
df=rate.find()
r=pd.DataFrame(list(df))
r=r.drop('_id',axis=1)
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import matplotlib.pyplot as plt
user_ids=r['UserID'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids=r["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
r["user"]=r["UserID"].map(user2user_encoded)
r["movie"]=r['movieId'].map(movie2movie_encoded)


num_users=len(user2user_encoded)
num_movies=len(movie_encoded2movie)
r['rating']=r['rating'].values.astype(np.float32)
min_rating=min(r['rating'])
max_rating=max(r['rating'])
print("Number of users: {}, Number of Movies:{},Min Rating:{},Max  Rating:{}".format(num_users,num_movies,min_rating,max_rating))


x=r[["user","movie"]].values
y=r["rating"].apply(lambda x:(x-min_rating)/(max_rating-min_rating)).values
train_indices=int(0.9*r.shape[0])
x_train,x_val,y_train,y_val=(
     x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],  
)


Model Creation
EMBEDDING_SIZE = 50




class RecommenderNet(keras.Model):
    def __init__(self, num_users, num_movies, embedding_size, **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_movies = num_movies
        self.embedding_size = embedding_size
        self.user_embedding = layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.user_bias = layers.Embedding(num_users, 1)
        self.movie_embedding = layers.Embedding(
            num_movies,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(1e-6),
        )
        self.movie_bias = layers.Embedding(num_movies, 1)


    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        movie_vector = self.movie_embedding(inputs[:, 1])
        movie_bias = self.movie_bias(inputs[:, 1])
        dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
        # Add all the components (including bias)
        x = dot_user_movie + user_bias + movie_bias
        # The sigmoid activation forces the rating to between 0 and 1
        return tf.nn.sigmoid(x)




model = RecommenderNet(num_users, num_movies, EMBEDDING_SIZE)
model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(),
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
)
history=model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=5,
    verbose=1,
    validation_data=(x_val,y_val)
)

df2=movies.find()
movie_df=pd.DataFrame(list(df2))
import json
movie_df['genres'] = movie_df['genres'].astype(str)
movie_df['genres'] = movie_df['genres'].str.replace("'", '"')
movie_df['genres'] = movie_df['genres'].apply(json.loads)
genre_names = []
for genres in movie_df['genres']:
    row_genre_names = [genre['name'] for genre in genres]
    genre_names.append(row_genre_names)


# Add genre names as a new column
movie_df['genre_names'] = genre_names


user_id= int(input("Enter user_id:"))
movies_watched_by_user=r[r.UserID==user_id]
movies_not_watched = movie_df[~movie_df["id"].isin(movies_watched_by_user["movieId"])]["id"]


movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]


user_encoder = user2user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movies_not_watched), movies_not_watched)
)
ratings = model.predict(user_movie_array).flatten()
top_ratings_indices = ratings.argsort()[-5:][::-1]
recommended_movie_ids = [
    movie_encoded2movie.get(movies_not_watched[x][0]) for x in top_ratings_indices
]
print("Showing recommendations for user: {}".format(user_id))
print("====" * 9)
print("Movies with high ratings from user")
print("----" * 8)
top_movies_user = (
    movies_watched_by_user.sort_values(by="rating", ascending=False)
    .head(5)
    .movieId.values
)
movie_df_rows = movie_df[movie_df["id"].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.original_title, ":", row.genre_names)


print("----" * 8)
print("Top 5 movies for you")
print("----" * 8)
recommended_movies = movie_df[movie_df["id"].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.original_title, ":", row.genre_names)
