from flask import Flask, render_template, request, flash, redirect, url_for, session,jsonify
from flask_session import Session
import finaltest
import redis


app = Flask(__name__)
app.secret_key = "secret key"


app.config['SESSION_TYPE'] = 'redis'
app.config['SESSION_PERMANENT'] = False
app.config['SESSION_USE_SIGNER'] = True
app.config['SESSION_REDIS'] = redis.from_url('redis://localhost:6379')


server_session = Session(app)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        genre = request.form.getlist('genre')
        if len(genre) >= 3:
            movies = finaltest.allmovierec(genre)
            title = list(movies['title'])
            movies['genres'] = movies['genres'].replace('    ',',', regex=True)
            genres = list(movies['genres'].replace('   ', '', regex=True))
            image = list(movies['image'])
            imdb_id=list(movies['imdb_id'])
            session['title'] = title
            session['genres'] = genres
            session['imdb_id']=imdb_id
            session['image']=image
            
            session.modified = True
            return redirect(url_for('recommendation'))
        else:           
            flash("Enter At Least 3 Genres!")
    return render_template('genreinput.html')


@app.route('/recommendation', methods=['POST', 'GET'])
def recommendation():
    title = session.get('title')
    image = session.get('image')
    genres = session.get('genres') 
    imdb_id=session.get('imdb_id')
    link=[]
    for i in imdb_id:
        i='https://www.imdb.com/title/'+str(i)
        link.append(i)
    if request.is_json:
        like_data = request.json
        title = like_data.get('title')
        rating = like_data.get('rating')
        print(rating)
    return render_template('genrerec.html', title=title, genres=genres, image=image,link=link)


if __name__ == '__main__':
    app.run(debug=True)



