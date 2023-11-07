import newspaper
import re
import nltk
import pandas as pd
import numpy as np
import datetime
import joblib
from datetime import datetime
from newspaper import news_pool
from newspaper import Article
from textblob import TextBlob
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
googlenews = GoogleNews()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()
from flask import Flask, request, jsonify
from flask_cors import CORS 
import mysql.connector

# Inisialisasi Flask
app = Flask(__name__)
CORS(app)

# Fungsi untuk menyimpan data dari DataFrame ke dalam MySQL
def save_df_to_mysql(dataframe, table_name, host, user, password, database):
    try:
        # Membuat koneksi ke database MySQL
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password,
            database=database
        )

        if connection.is_connected():
            cursor = connection.cursor()

            # Loop melalui setiap baris dalam DataFrame
            for index, row in dataframe.iterrows():
                # Query untuk menyimpan data ke dalam tabel MySQL
                insert_query = f"""
                INSERT INTO {table_name} (judul, penulis, text, summary, date, source, translate, sentiment, svm)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
                # Eksekusi query dengan data dari setiap baris DataFrame
                data_to_insert = (row['judul'], row['penulis'], row['text'], row['summary'], row['date'], row['source'], row['translate'], row['sentiment'], row['svm'])
                cursor.execute(insert_query, data_to_insert)

            # Commit perubahan ke database
            connection.commit()
            print("Data berhasil disimpan ke MySQL!")

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Fungsi untuk mendapatkan berita dari Google News
def get_google_news(figure_name):
    combined_data = []
    googlenews.clear()
    googlenews.enableException(True)
    googlenews.set_lang('id')
    googlenews.set_period('7d')
    googlenews.set_encode('utf-8')
    googlenews.search(figure_name)

    for x in range(5):
        result = googlenews.page_at(x)
        links = [item["link"] for item in result]
        filtered_data = [re.sub(r'&ved.*', '', item) for item in links]
        combined_data.extend(filtered_data)

    words_to_remove = ['youtube', 'google', 'tribunnews', 'suaramerdeka',
                       'jawapos', 'ayojakarta', 'rilis.id', 'kilat.com',
                       'pojoksatu', 'maharnews', 'mengerti.id', 'rmolsumsel',
                       'intipseleb', 'harianreportase', 'kompas.tv']

    list_of_urls = [item for item in combined_data if all(word not in item for word in words_to_remove)]
    return list_of_urls

# Load the saved pkl model and vectorizers
svm_model = joblib.load('svm_sentiment_model.pkl')
vectorizer = joblib.load('svm_tfidf_vectorizer.pkl')

# Preprocess the text
def preprocess_text(text, use_stemming=True):
    text = re.sub(r'\W+', ' ', text).lower()
    tokens = word_tokenize(text)
    if use_stemming:
        filtered_tokens = [stemmer.stem(w) for w in tokens if not w in stop_words]
    else:
        filtered_tokens = [w for w in tokens if not w in stop_words]
    return ' '.join(filtered_tokens)

def predict_sentiment(text):
    preprocessed_text = preprocess_text(text, use_stemming=True)
    text_vector = vectorizer.transform([preprocessed_text])
    sentiment = svm_model.predict(text_vector)[0]
    return sentiment 

# Fungsi untuk menerjemahkan teks
def translation(text):
    try:
        blob = TextBlob(text)
        return str(blob.translate(from_lang='id', to='en'))
    except Exception as e:
        print(f"Translation error: {str(e)}")
        return text

# Fungsi untuk analisis sentimen menggunakan TextBlob
def sentimentblob(text):
    try:
        blob = TextBlob(text)
        return blob.sentiment.polarity
    except Exception as e:
        print(f"Sentiment analysis error: {str(e)}")
        return 0.0

# Endpoint untuk mengambil dan menyimpan berita ke database
@app.route('/fetch_and_save_news', methods=['POST'])
def fetch_and_save_news():
    try:
        data = request.get_json()
        figure_name = data['figure_name']
        list_of_urls = get_google_news(figure_name)

        df_news = pd.DataFrame()

        for url in list_of_urls:
            try:
                url_i = newspaper.Article(url="%s" % (url), language='id')
                url_i.download()
                url_i.parse()
                url_i.nlp()

                df = pd.DataFrame(columns=['judul', 'penulis', 'text', 'summary', 'date', 'source',
                                           'translate', 'sentiment', 'svm'])
                sentiments = SentimentIntensityAnalyzer()

                df['penulis'] = url_i.authors
                df['judul'] = url_i.title
                df['text'] = url_i.text
                df['summary'] = url_i.summary
                df['translate'] = df['summary'].apply(translation)
                df['date'] = url_i.publish_date
                df['source'] = url_i.url
                sentimentBlob = df['translate'].apply(sentimentblob)
                df['sentiment'] = np.where(sentimentBlob < 0, "Negative",
                                            np.where(sentimentBlob > 0, "Positive", "Neutral"))
                df['svm'] = df['summary'].apply(predict_sentiment)
                df.loc[(df['sentiment'] == 'Neutral') & (df['svm'] == 'Positive'), 'svm'] = 'Neutral'

                df_news = pd.concat([df_news, df], ignore_index=True)
                df_dt = df_news.drop_duplicates(subset=['judul'])

            except Exception as e:
                print(f"Error processing URL {url}: {str(e)}")

        # Simpan data ke MySQL
        save_df_to_mysql(df_dt, 'news_data', 'localhost', 'root', '', 'skripsi')

        return jsonify({'message': 'Data berhasil diambil dan disimpan ke MySQL!'})

    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/analyze_url', methods=['POST'])
def analyze_url():
    try:
        url = request.json['url']

        rows = []
        a = Article(url, language='id')
        a.download()
        a.parse()
        a.nlp()

        url = a.url
        title = a.title
        author = a.authors
        date = a.publish_date
        text = a.text
        summary = a.summary
        image = a.top_image
        keyword = a.keywords
        source = a.source_url

        translate = translation(summary)
        sentimentScore = sentimentblob(translate)
        sentimentLabel = np.where(sentimentScore < 0, "Negative",
                                  np.where(sentimentScore > 0, "Positive", "Netral"))

        row = {'url': url, 'title': title, 'author': author[0], 'date': date, 'text': text, 'sentimentScore': sentimentScore,
               'sentimentLabel': str(sentimentLabel), 'image': image, 'keyword': keyword, 'summary': summary, 'source': source}
        rows.append(row)

        df_news = pd.DataFrame(rows)

        return jsonify(df_news.to_dict(orient='records'))

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
