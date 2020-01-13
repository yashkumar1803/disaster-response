import json
import plotly
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from collections import Counter
import numpy as np


from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Heatmap
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier-v4.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # created a visual to visualize the top 10 words in the disaster response
    # dataset. This does not include stop words
    text = " ".join(list(df["message"]))
    lemmatizer = WordNetLemmatizer()
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    text_list = text.replace("""'""", "").lower().translate(translator).split()

    word_frequency = Counter(text_list)

    for word in stopwords.words("english"):
        word_frequency.pop(word, None)

    for word in list(word_frequency.keys())[:]:
        if word_frequency[word] == 1:
            word_frequency.pop(word, None)
            
    x_words, y_frequencies = list(map(list, zip(*word_frequency.most_common(10))))
    
    # created a categorical heatmap to observe how the sentence length changes between the different message genres (direct, social and news)
    df["sentence_length"] = df["message"].apply(lambda x: len(x.replace("""'""", "").lower().translate(translator).split()))

    categories = list(df.columns[4:-1])
    genres = list(df["genre"].unique())

    category_dict = dict(zip(categories, range(len(categories))))
    genre_dict = dict(zip(genres, range(len(categories))))

    arr = np.zeros((len(genres), len(categories)))

    for category in categories:
        for genre in genres:
            
            arr[genre_dict[genre], category_dict[category]] = np.median(list(df[(df[category]==1) & (df["genre"]==genre)]["sentence_length"]))

    # pd.DataFrame(arr, columns=genres, index=categories).T
    
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        
        {
            'data': [
                Bar(
                    x = x_words,
                    y = y_frequencies
                
                )
            ],

            'layout': {
                'title': 'Most frequent words in messages (excluding stopwords)',
                'yaxis': {
                    'title': "Frequency"
                },
                'xaxis': {
                    'title': "Common Words"
                }
            }            
        },
        
        {
            'data': [
                Heatmap(
                    
                    z = arr,
                    x = [word.replace("_", " ") for word in categories],
                    y = genres,
                    colorscale = "RdYlGn_r"
                
                )
            ],

            'layout': {
                'title': 'Message Length Distribution',
                'yaxis': {
                    'title': "Genres"
                },
                'xaxis': {
                    #'title': "Categories"
                }
            }            
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[3:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
