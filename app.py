# app.py
from flask import Flask, render_template, request
import pandas as pd
from scipy import spatial
# from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

app = Flask(__name__)

# Load data yang sudah diproses
df = pd.read_csv('webtoon_originals_id.csv')  # Ubah ke file yang sesuai
features = df[['likes', 'rating']]

# Load scaler
scaler = MinMaxScaler(feature_range=(0, 1), copy=True)
features_scaled = scaler.fit_transform(features)

# Fungsi untuk mendapatkan webtoon teratas dari DataFrame
def get_top_webtoons():
    top_webtoons = df.sort_values(by=['likes', 'rating'], ascending=[False, False]).head(10)
    top_webtoons['image_filename'] = ['nobless.jpg', 'lookism.jpg','truebeauty.jpg','tahilalats.jpg',
                                       'ngopiyuk.jpg', 'pasutrigj.jpg', 'tog.jpg', 
                                       'eggnoid.jpg','flawless.jpg', 'change.jpg', ]

    # Cetak DataFrame untuk memeriksa keberadaan kolom
    print(top_webtoons)

    # Ubah ke dalam bentuk list of dictionaries
    top_webtoons_list = top_webtoons.to_dict('records')[:10]

    return top_webtoons_list

# Route untuk halaman utama
@app.route('/')
def index():
    top_webtoons = get_top_webtoons()
    return render_template('index.html', top_webtoons=top_webtoons)


@app.route('/recommend', methods=['POST'])
def recommend():
    input_likes = float(request.form['likes'])
    input_rating = float(request.form['rating'])
    input_genre = request.form['genre']

    # Normalisasi input pengguna
    input_data = scaler.transform([[input_likes, input_rating]])

    # Hitung kesamaan dengan setiap data di DataFrame
    df['similarity'] = df.apply(lambda row: 1 - spatial.distance.cosine(input_data.flatten(), scaler.transform([[row['likes'], row['rating']]]).flatten()), axis=1)

    # df['similarity'] = df.apply(lambda row: 1 - spatial.distance.cosine(input_data, scaler.transform([[row['likes'], row['rating']]])), axis=1)

    # Urutkan DataFrame berdasarkan kesamaan
    recommendations = df.sort_values(by='similarity', ascending=False)

    # Filter berdasarkan genre jika input_genre tidak None
    if input_genre:
        recommendations = recommendations[recommendations['genre'] == input_genre]
    
    return render_template('index.html', recommendations=recommendations[['title', 'genre']].values)


if __name__ == '__main__':
    app.run(debug=True)
