from flask import Flask, render_template, request, redirect,flash,url_for,session,jsonify
from flask_session import Session
import finaltest
import redis
from pymongo import MongoClient
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
import requests
from bs4 import BeautifulSoup
from collections import Counter
app = Flask(__name__)
client = MongoClient('mongodb://localhost:27017/')
db = client['Movies']
users_collection = db['User']
print(client.list_database_names())
app.secret_key = "secret key"

app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')
server_session=Session(app)
# Check MongoDB connection
def check_db_connection():
    try:
        client.server_info()
        print('Connected to MongoDB successfully!')
    except Exception as e:
        print('Failed to connect to MongoDB:', str(e))

check_db_connection()
client = MongoClient("localhost", 27017)
db = client['Movies']
import pandas as pd
rate = db['Ratings_Small']
movies = db["meta"]
df2 = movies.find()
movie_df = pd.DataFrame(list(df2))
df = rate.find()
r = pd.DataFrame(list(df))
r = r.drop('_id', axis=1)
print(r.head(5))
user_ids = r['UserID'].unique().tolist()
user2user_encoded = {x: i for i, x in enumerate(user_ids)}
userencoded2user = {i: x for i, x in enumerate(user_ids)}
movie_ids = r["movieId"].unique().tolist()
movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
movie_encoded2movie = {i: x for i, x in enumerate(movie_ids)}
r["user"] = r["UserID"].map(user2user_encoded)
r["movie"] = r['movieId'].map(movie2movie_encoded)
num_users = len(user2user_encoded)
num_movies = len(movie_encoded2movie)
r['rating'] = r['rating'].values.astype(np.float32)
min_rating = min(r['rating'])
max_rating = max(r['rating'])
r = r.sample(frac=1, random_state=42)
x = r[["user", "movie"]].values
y = r["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
train_indices = int(0.9 * r.shape[0])
x_train, x_val, y_train, y_val = (
    x[:train_indices],
    x[train_indices:],
    y[:train_indices],
    y[train_indices:],
)
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
    run_eagerly=True
)
history = model.fit(
    x=x_train,
    y=y_train,
    batch_size=64,
    epochs=5,
    verbose=1,
    validation_data=(x_val, y_val)
)
def scrape_wikipedia_data(movie_title):
    # Construct the Wikipedia URL for the movie title
    wikipedia_url = f"https://en.wikipedia.org/wiki/{movie_title.replace(' ', '_')}"

    try:
        # Send an HTTP request to get the Wikipedia page content
        response = requests.get(wikipedia_url)
        response.raise_for_status()  # Raise an exception for 4xx and 5xx status codes

        # Parse the HTML content using BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find the infobox table
        infobox = soup.find("table", class_="infobox")

        if infobox:
            # Find the img tag within the infobox (poster URL)
            poster_element = infobox.find("img")
            poster_url = poster_element['src'] if poster_element else ""

            # Find the summary text from the first paragraph in the infobox
            summary_element = infobox.find("p")
            summary = summary_element.get_text() if summary_element else ""

            # Extract the release year from the infobox
            release_year_element = infobox.find("span", class_="bday dtstart published updated")
            release_year = release_year_element.get_text() if release_year_element else ""

            return {
                "summary": summary,
                "release_year": release_year,
                "poster_url": poster_url
            }
        else:
            print(f"No infobox found for {movie_title}.")
            return None

    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None
    except Exception as err:
        print(f"Error occurred: {err}")
        return None


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
   if request.method == 'POST':
        UserID = request.form.get('UserID')
        Password = request.form.get('Password')
        print('Entered UserID:', UserID)
        print('Entered Password:', Password)

        if not UserID:
            return 'Please enter a valid UserID.'

        try:
            UserID = int(UserID)
        except ValueError:
            return 'Invalid UserID. UserID must be a number.'

        user = users_collection.find_one({'UserID': UserID, 'Password': Password})
        print('User from DB:', user)

        if user:
            # Redirect to a different page on successful login
            return redirect(f'/success?UserID={UserID}')
        else:
            return 'Invalid userID or Password.'

   if request.method == 'GET':
        return render_template('login.html')
    
@app.route('/success')
def success():
    user_id = int(request.args.get('UserID'))
    print("Recieved UserID:",user_id)
    print(r[r.UserID == user_id])
    movies_watched_by_user = r[r.UserID == user_id+1]
    watched_movie_ids = movies_watched_by_user["movieId"].tolist()
    print(watched_movie_ids)
    if movies_watched_by_user.empty:
        return 'No Movies watched by user.'
    watched_movie_data = []
    watched_movies_df = movie_df[movie_df["id"].isin(watched_movie_ids)]
    print(watched_movies_df)

# Loop through the filtered DataFrame to extract movie information
    count = 0  # Initialize the counter
    for _, movie_info in watched_movies_df.iterrows():
      if count >= 5:
        break  # Break the loop once 5 movies have been added
      movie_title = movie_info["original_title"]
      movie_info_data = scrape_wikipedia_data(movie_title)

      if movie_info_data:
        watched_movie_data.append({
            "title": movie_title,
            "summary": movie_info_data["summary"],
            "release_year": movie_info_data["release_year"],
            "poster_url": movie_info_data["poster_url"]
        })
        count += 1  # Increment the counter for each movie added


    movies_not_watched = movie_df[~movie_df["id"].isin(movies_watched_by_user["movieId"])]["id"]
    movies_not_watched = list(set(movies_not_watched).intersection(set(movie2movie_encoded.keys())))
    if not movies_not_watched:
        return 'You watched everything or what'
    movies_not_watched = [
        x for x in movies_not_watched
        if movie_df[movie_df["id"] == x]["poster_path"].values[0] is not None]
    movies_not_watched = [movie2movie_encoded[x] for x in movies_not_watched if x in movie2movie_encoded]
    user_encoder = user2user_encoded.get(user_id)
    user_movie_list = []
    for x in movies_not_watched:
        user_movie_list.append([user_encoder, x])

    user_movie_array = np.array([x for x in user_movie_list if None not in x])
    if user_movie_array.size == 0:
        return 'No movies available for recommendation.'
    tf.config.run_functions_eagerly(True)
    ratings = model.predict(user_movie_array).flatten()
    top_ratings_indices = ratings.argsort()[-5:][::-1]
    recommended_movie_ids = [movies_not_watched[x] for x in top_ratings_indices]
    recommended_movies = movie_df[movie_df["id"].isin(recommended_movie_ids)]

    # Create a list of movie IDs and corresponding poster URLs
    movie_data = []
    for row in recommended_movies.itertuples():
        tmdb_id = row.id
        movie_title = movie_df[movie_df["id"] == tmdb_id]["original_title"].values[0]
        movie_info = scrape_wikipedia_data(movie_title)
        if movie_info:
            movie_data.append({
                "title": movie_title,
                "summary": movie_info["summary"],
                "release_year": movie_info["release_year"],
                "poster_url": movie_info["poster_url"]
            })

    return render_template('oldrec.html', movie_data=movie_data,watched_movie_data=watched_movie_data)

# Run the test function
import pandas as pd
from itertools import combinations
client = MongoClient('mongodb://localhost:27017/')
db = client['Movies']
collection=db['meta']
cursor=collection.find()
data=list(cursor)
movies=pd.DataFrame(data)
movies['genres']=movies['genres'].astype('str')

def ratings(userid,id,rating):
    db=client['Movies']
    collection=db['Ratings_Small']
    cursor=collection.find()
    data=list(cursor)
    ratingsdata=pd.DataFrame(data)
    data = ratingsdata[ratingsdata['UserID']==str(userid)]
    if str(id) in str(data.movieId) and str(userid) in str(ratingsdata['UserID']):
        if rating==0:
             collection.delete_one({'UserID':str(userid),'movieId':str(id)})
        else:
            collection.update_one({'movieId':str(id)}, {"$set": {'UserID':str(userid),'movieId':str(id),'rating':int(rating),'timestamp':0}})
    else:
        collection.insert_one({'UserID':str(userid),'movieId':str(id),'rating':int(rating),'timestamp':0})


def movierec(genrelist,ids,n_rec):
    recmovielist=Counter()
    
    for genre in genrelist:
        recmovielist += Counter(movies[movies['genres'].str.contains(genre)]['id'].tolist())

    recmovies = [movie_id for movie_id, genre_count in recmovielist.items() if genre_count >= len(genrelist)]
    rec_moviesdata = movies[movies['id'].isin(recmovies)]
    rec_moviesdata=movies[movies['id'].isin(recmovies)]
    rec_moviesdata=rec_moviesdata[~rec_moviesdata['id'].isin(ids)]
    rec_moviesdata=rec_moviesdata.sort_values(by='popularity',ascending=False)
    return rec_moviesdata.head(10//n_rec+1)

def allmovierec(genrelist, n_rec):
    ids = []
    genrelist = [genrelist] if not isinstance(genrelist, list) else genrelist
    allrec_moviesdata = pd.DataFrame(columns=['id'])
    for i in range(1, len(genrelist) + 1):   
        for genres in combinations(genrelist, i):
            rec_moviesdata = movierec(genres, ids, len(genrelist))
            allrec_moviesdata = pd.concat([allrec_moviesdata, rec_moviesdata])
            for i in range(0, allrec_moviesdata.shape[0]):
                if allrec_moviesdata['id'].iloc[i] not in ids:
                    ids.append(allrec_moviesdata['id'].iloc[i])
    allrec_moviesdata = allrec_moviesdata.sort_values(by='popularity', ascending=False)
    image = []
    for i in allrec_moviesdata['title']:
        i = str(i)
        i = i.replace(' ', '_')
        i = i.replace(':', '')
        i = i.replace('-', '')
        i = i.replace('.', '')
        i = i.replace(',', '')
        url = ('https://www.rottentomatoes.com/m/' + str(i) + '')
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        infobox = soup.find('div', {'class': 'movie-thumbnail-wrap'})
        if 'img' in str(infobox):
            image.append(infobox.img.attrs['src'])
        else:
            image.append('https://www.wildhareboca.com/wp-content/uploads/sites/310/2018/03/image-not-available-200x300.jpg')
    allrec_moviesdata['image'] = image
    return allrec_moviesdata.head(n_rec).sample(n_rec)


def signupdb(userid,password,password_confirmation):
    db=client['Movies']
    collection=db['User']
    data=list(collection.find({'UserID': userid}))
    userdetails=pd.DataFrame(data)
    print(userdetails)
    if not userdetails.empty and (userid in str(userdetails['UserID'].values)):
        return 0
    else:
        if len(password) >= 8 and password == password_confirmation:
            collection.insert_one({'UserID': userid, 'Password': password})
            return 1
        return -1
    
@app.route('/signup',methods=['GET','POST'])
def signup():
     if request.method == 'POST':
        UserID = request.form.get('UserID')
        Password = request.form.get('Password')
        password_confirmation = request.form.get('Password_confirmation')

        if signupdb(UserID, Password, password_confirmation) == 1:
            if Password != password_confirmation:
                flash('The password doesnt match')
            elif len(Password) < 8:
                flash('The password must be at least 8 characters')
            else:
                session['userid'] = UserID
                return redirect(url_for('genreinput'))
        else:
            flash('The User ID already exists')

     return render_template('signup.html')


@app.route('/genreinput', methods=['GET', 'POST'])
def genreinput():
    if request.method == 'POST':
        genre = request.form.getlist('genre')
        if len(genre) >= 3:
            movies = allmovierec(genre,len(genre))
            title = list(movies['title'])
            movies['genres'] = movies['genres'].replace('    ',',', regex=True)
            genres = list(movies['genres'].replace('   ', '', regex=True))
            image = list(movies['image'])
            imdb_id=list(movies['imdb_id'])
            id=list(movies['id'])
            session['title'] = title
            session['genres'] = genres
            session['imdb_id']=imdb_id
            session['image']=image
            session['id']=id
            
            session.modified = True
            return redirect(url_for('recommendation'))
        else:           
            flash("Enter At Least 3 Genres!")
    return render_template('genreinput.html')

@app.route('/recommendation', methods=['GET','POST'])
def recommendation():
    # Check if the form is submitted via POST request
    if request.method == 'POST':
        # Get the selected genres from the form
        selected_genres = request.form.getlist('genre')
        
        # Check if at least 3 genres are selected
        if len(selected_genres) >= 3:
            # Call the allmovierec function to get movie recommendations based on the selected genres
            movies = allmovierec(selected_genres)
            
            # Extract movie details from the DataFrame
            title = list(movies['title'])
            genres = list(movies['genres'])
            image = list(movies['image'])
            imdb_id = list(movies['imdb_id'])
            
            # Store the movie details in the session for later use
            session['title'] = title
            session['genres'] = genres
            session['image'] = image
            session['imdb_id'] = imdb_id
            
            # Redirect to the 'genrerec.html' template to display movie recommendations
            return redirect(url_for('recommendation'))
        else:
            flash("Enter At Least 3 Genres!")
    
    # Get the movie details from the session
    title = session.get('title')
    image = session.get('image')
    genres = session.get('genres') 
    imdb_id = session.get('imdb_id')
    userid = session.get('userid')
    
    # Generate IMDb links for the movie recommendations
    link = ['https://www.imdb.com/title/' + str(i) for i in imdb_id]
    
    if request.is_json:
        data = request.json
        movie_id = data.get('id')
        rating = data.get('rating')
        print(movie_id)
        print(rating)
        # Call the ratings function to store the user ratings
        ratings(userid, movie_id, rating)
    
    # Render the 'genrerec.html' template with the movie recommendations
    return render_template('genrerec.html', title=title, genres=genres, image=image, link=link, id=id)
    

    

if __name__ == '__main__':
    app.run(debug=True)