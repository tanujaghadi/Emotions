from __future__ import division, print_function
import cv2
import numpy as np
import tensorflow as tf
import statistics as st
from flask import Flask, render_template, request, redirect, url_for, session
import json
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from flask_sqlalchemy import SQLAlchemy

with open('config.json', 'r') as c:
    params = json.load(c)["params"]

app = Flask(__name__)
app.secret_key = 'secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = params["local_uri"]
db = SQLAlchemy(app)


class Userinfo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(50), unique=False, nullable=False)
    password = db.Column(db.String(5), unique=False, nullable=False)


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        '''Add entry to the database'''
        name = request.form.get('name')
        email = request.form.get('email')
        password = request.form.get('password')
        entry = Userinfo(name=name, email=email, password=password)
        db.session.add(entry)
        db.session.commit()
        return redirect(url_for('successful'))  # Redirect to login page

    return render_template('register.html', params=params)


@app.route("/example")
def example():
    return render_template("example.html")


@app.route("/successful")
def successful():
    return render_template("successful.html")


@app.route('/')
def homepage():
    return render_template('login.html')


@app.route('/home')
def home():
    return render_template('index1.html')


# Base = declarative_base()
# engine = create_engine('mysql://root:@localhost/users', echo=True)  # Replace with your database URL
# Base.metadata.create_all(engine)  # Create the tables in the database
# Session = sessionmaker(bind=engine)
# session1 = Session()
#
# user_data = session1.query(Userinfo).all()
# user_data_table = {}
# for user in user_data:
#     user_data_table[user.name] = user.password
# print(user_data_table)


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('name')
        password = request.form.get('password')
        user1 = Userinfo.query.filter_by(name=username).first()
        passw = Userinfo.query.filter_by(password=password).first()
        if user1 and passw:
            # if username in user_data_table.keys() and password == user_data_table.get(username):
            return redirect(url_for('home'))

    return render_template('login.html', params=params)


@app.route('/camera', methods=['GET', 'POST'])
def camera():
    i = 0

    GR_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
    model = tf.keras.models.load_model('final_model.h5')
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    output = []
    cap = cv2.VideoCapture(0)
    while (i <= 30):
        ret, img = cap.read()
        faces = face_cascade.detectMultiScale(img, 1.05, 5)

        for x, y, w, h in faces:
            face_img = img[y:y + h, x:x + w]

            resized = cv2.resize(face_img, (224, 224))
            reshaped = resized.reshape(1, 224, 224, 3) / 255
            predictions = model.predict(reshaped)

            max_index = np.argmax(predictions[0])

            emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'neutral', 'surprise')
            predicted_emotion = emotions[max_index]
            output.append(predicted_emotion)

            cv2.rectangle(img, (x, y), (x + w, y + h), GR_dict[1], 2)
            cv2.rectangle(img, (x, y - 40), (x + w, y), GR_dict[1], -1)
            cv2.putText(img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        i = i + 1

        cv2.imshow('LIVE', img)
        key = cv2.waitKey(1)
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            break
    print(output)
    cap.release()
    cv2.destroyAllWindows()
    final_output1 = st.mode(output)
    return render_template("buttons.html", final_output=final_output1)


@app.route('/templates/buttons', methods=['GET', 'POST'])
def buttons():
    return render_template("buttons.html")


@app.route('/movies/surprise', methods=['GET', 'POST'])
def moviesurprise():
    return render_template("moviesSurprise.html")


@app.route('/movies/angry', methods=['GET', 'POST'])
def moviesAngry():
    return render_template("moviesAngry.html")


@app.route('/movies/sad', methods=['GET', 'POST'])
def moviesSad():
    return render_template("moviesSad.html")


@app.route('/movies/disgust', methods=['GET', 'POST'])
def moviesDisgust():
    return render_template("moviesDisgust.html")


@app.route('/movies/happy', methods=['GET', 'POST'])
def moviesHappy():
    return render_template("moviesHappy.html")


@app.route('/movies/fear', methods=['GET', 'POST'])
def moviesFear():
    return render_template("moviesFear.html")


@app.route('/movies/neutral', methods=['GET', 'POST'])
def moviesNeutral():
    return render_template("moviesNeutral.html")


@app.route('/songs/surprise', methods=['GET', 'POST'])
def songsSurprise():
    return render_template("songsSurprise.html")


@app.route('/songs/angry', methods=['GET', 'POST'])
def songsAngry():
    return render_template("songsAngry.html")


@app.route('/songs/sad', methods=['GET', 'POST'])
def songsSad():
    return render_template("songsSad.html")


@app.route('/songs/disgust', methods=['GET', 'POST'])
def songsDisgust():
    return render_template("songsDisgust.html")


@app.route('/songs/happy', methods=['GET', 'POST'])
def songsHappy():
    return render_template("songsHappy.html")


@app.route('/songs/fear', methods=['GET', 'POST'])
def songsFear():
    return render_template("songsFear.html")


@app.route('/songs/neutral', methods=['GET', 'POST'])
def songsNeutral():
    return render_template("songsSad.html")


if __name__ == "__main__":
    app.run(debug=True)
